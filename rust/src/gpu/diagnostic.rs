//! Diagnostic pipeline for canonical angular momentum conservation test.
//!
//! Traces a small number of particles in a uniform analytic Bz field and
//! records (x, y, z, ux, uy, uz) at every step. The caller post-processes
//! the CSV to verify P_φ = m(x·uy − y·ux) + (q·Bz/2)(x²+y²) is conserved.
//!
//! Nothing here touches the main run pipeline.

use ash::vk;
use anyhow::{Context, Result};
use bytemuck::Zeroable;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc, Allocation,
                             AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::ffi::CString;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::VulkanContext;

// ── C-layout structs matching the GLSL shader ────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InitState {
    x: f32, y: f32, z: f32, _p0: f32,
    ux: f32, uy: f32, uz: f32, _p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StepRecord {
    x: f32, y: f32, z: f32,
    ux: f32, uy: f32, uz: f32,
    step: u32, _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DiagPushConst {
    dt: f32,
    q_over_m: f32,
    n_particles: u32,
    n_steps: u32,
    bz: f32,
    _pad: [u32; 3],
}

// ── Simple owned buffer ───────────────────────────────────────────────────────

struct OwnedBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
}

impl OwnedBuffer {
    unsafe fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let reqs = device.get_buffer_memory_requirements(buffer);
        let allocation = allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name,
            requirements: reqs,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        Ok(Self { buffer, allocation: Some(allocation) })
    }

    fn write_bytes(&self, data: &[u8]) {
        let alloc = self.allocation.as_ref().unwrap();
        let ptr = alloc.mapped_ptr().expect("Buffer not CPU-visible").as_ptr() as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()); }
    }

    fn read_bytes(&self, dst: &mut [u8]) {
        let alloc = self.allocation.as_ref().unwrap();
        let ptr = alloc.mapped_ptr().expect("Buffer not CPU-visible").as_ptr() as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()); }
    }

    unsafe fn destroy(mut self, device: &ash::Device, allocator: &Arc<Mutex<Allocator>>) {
        device.destroy_buffer(self.buffer, None);
        if let Some(a) = self.allocation.take() {
            allocator.lock().unwrap().free(a).ok();
        }
    }
}

// ── Public entry point ───────────────────────────────────────────────────────

/// Run the diagnostic Boris shader and write a CSV of per-step particle states.
///
/// `init`: (x_m, y_m, z_m, ux_ms, uy_ms, uz_ms) initial conditions.
/// `bz`:   uniform Bz [T], analytic — no field texture needed.
/// `dt_s`: timestep [s].
pub fn run_canonical_diag(
    ctx: &VulkanContext,
    init: &[(f32, f32, f32, f32, f32, f32)],
    bz: f32,
    dt_s: f32,
    n_steps: u32,
    out_csv: &Path,
) -> Result<()> {
    assert!(!init.is_empty() && n_steps > 0);
    let spv = include_bytes!("../../../shaders/boris_diag.spv");
    unsafe { run_inner(ctx, spv, init, bz, dt_s, n_steps, out_csv) }
}

unsafe fn run_inner(
    ctx: &VulkanContext,
    spv: &[u8],
    init: &[(f32, f32, f32, f32, f32, f32)],
    bz: f32,
    dt_s: f32,
    n_steps: u32,
    out_csv: &Path,
) -> Result<()> {
    let device = ctx.device();
    let n_particles = init.len() as u32;

    // ── Allocator ────────────────────────────────────────────────────────────
    let allocator = Arc::new(Mutex::new(
        Allocator::new(&AllocatorCreateDesc {
            instance: ctx.instance().clone(),
            device: device.clone(),
            physical_device: ctx.physical_device(),
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).context("Failed to create allocator for diagnostic")?,
    ));

    // ── Shader module ────────────────────────────────────────────────────────
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))
        .context("Failed to parse diagnostic SPIR-V")?;
    let module = device.create_shader_module(
        &vk::ShaderModuleCreateInfo::default().code(&code), None,
    )?;

    // ── Descriptor set layout ────────────────────────────────────────────────
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let dsl = device.create_descriptor_set_layout(
        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None,
    )?;

    // ── Pipeline layout ───────────────────────────────────────────────────────
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0).size(std::mem::size_of::<DiagPushConst>() as u32);
    let pipeline_layout = device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&dsl))
            .push_constant_ranges(std::slice::from_ref(&push_range)),
        None,
    )?;

    // ── Compute pipeline ──────────────────────────────────────────────────────
    let entry = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry);
    let pipeline = device.create_compute_pipelines(
        vk::PipelineCache::null(),
        &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
        None,
    ).map_err(|(_, e)| anyhow::anyhow!("Diagnostic pipeline creation: {:?}", e))?[0];

    // ── Descriptor pool + set ─────────────────────────────────────────────────
    let pool = device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(2),
        ]), None,
    )?;
    let descriptor_set = device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool).set_layouts(std::slice::from_ref(&dsl)),
    )?[0];

    // ── Buffers ───────────────────────────────────────────────────────────────
    let init_size  = (n_particles as usize * std::mem::size_of::<InitState>()) as u64;
    let trace_size = (n_particles as usize * n_steps as usize * std::mem::size_of::<StepRecord>()) as u64;

    let init_buf = OwnedBuffer::new(device, &allocator, init_size,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "diag_init")?;
    let trace_buf = OwnedBuffer::new(device, &allocator, trace_size,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu, "diag_trace")?;

    // Write initial conditions
    let init_data: Vec<InitState> = init.iter().map(|&(x,y,z,ux,uy,uz)| InitState {
        x, y, z, _p0: 0.0, ux, uy, uz, _p1: 0.0,
    }).collect();
    init_buf.write_bytes(bytemuck::cast_slice(&init_data));

    // ── Bind buffers ──────────────────────────────────────────────────────────
    device.update_descriptor_sets(&[
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set).dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&[vk::DescriptorBufferInfo::default()
                .buffer(init_buf.buffer).offset(0).range(init_size)]),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set).dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&[vk::DescriptorBufferInfo::default()
                .buffer(trace_buf.buffer).offset(0).range(trace_size)]),
    ], &[]);

    // ── Record and submit ─────────────────────────────────────────────────────
    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1),
    )?[0];

    device.begin_command_buffer(cmd,
        &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
        pipeline_layout, 0, &[descriptor_set], &[]);

    let push = DiagPushConst {
        dt: dt_s,
        q_over_m: 9.578_701_5e7_f32,
        n_particles,
        n_steps,
        bz,
        _pad: [0; 3],
    };
    device.cmd_push_constants(cmd, pipeline_layout,
        vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));

    device.cmd_dispatch(cmd, n_particles, 1, 1);

    device.cmd_pipeline_barrier(cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::HOST,
        vk::DependencyFlags::empty(), &[], &[
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .buffer(trace_buf.buffer).offset(0).size(trace_size),
        ], &[]);

    device.end_command_buffer(cmd)?;

    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    device.queue_submit(ctx.compute_queue(),
        &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))], fence)?;
    device.wait_for_fences(&[fence], true, u64::MAX)?;

    // ── Read back ─────────────────────────────────────────────────────────────
    let mut records = vec![StepRecord::zeroed(); n_particles as usize * n_steps as usize];
    trace_buf.read_bytes(bytemuck::cast_slice_mut(&mut records));

    write_csv(out_csv, &records, n_particles, n_steps)?;

    // ── Cleanup ───────────────────────────────────────────────────────────────
    device.destroy_fence(fence, None);
    device.free_command_buffers(ctx.command_pool(), &[cmd]);
    init_buf.destroy(device, &allocator);
    trace_buf.destroy(device, &allocator);
    device.destroy_descriptor_pool(pool, None);
    device.destroy_pipeline(pipeline, None);
    device.destroy_pipeline_layout(pipeline_layout, None);
    device.destroy_descriptor_set_layout(dsl, None);
    device.destroy_shader_module(module, None);

    log::info!("Diagnostic complete → {:?}", out_csv);
    Ok(())
}

fn write_csv(path: &Path, records: &[StepRecord], n_particles: u32, n_steps: u32) -> Result<()> {
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("Cannot create {:?}", path))?
    );
    writeln!(f, "particle,step,x_m,y_m,z_m,ux_ms,uy_ms,uz_ms")?;
    for pid in 0..n_particles as usize {
        for s in 0..n_steps as usize {
            let r = &records[pid * n_steps as usize + s];
            writeln!(f, "{},{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
                pid, r.step, r.x, r.y, r.z, r.ux, r.uy, r.uz)?;
        }
    }
    Ok(())
}
