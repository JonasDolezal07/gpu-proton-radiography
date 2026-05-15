//! Orbit camera for 3D navigation
//!
//! Controls:
//! - Left mouse drag: orbit (rotate around target)
//! - Right mouse drag: pan (move target)
//! - Scroll wheel: zoom (change distance)

use glam::{Mat4, Vec3};

pub struct Camera {
    // Spherical coordinates around target
    distance: f32,
    yaw: f32,   // Horizontal angle (radians)
    pitch: f32, // Vertical angle (radians)
    target: Vec3,

    // Cached matrices
    view: Mat4,
    proj: Mat4,
    view_proj: Mat4,
    inv_view_proj: Mat4,

    // Window dimensions
    width: u32,
    height: u32,

    // Input state
    is_orbiting: bool,
    is_panning: bool,
    last_mouse_pos: Option<(f32, f32)>,
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        let mut camera = Self {
            distance: 1.0,
            yaw: 0.5,        // ~30 degrees
            pitch: 0.3,      // ~17 degrees
            target: Vec3::ZERO,
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            view_proj: Mat4::IDENTITY,
            inv_view_proj: Mat4::IDENTITY,
            width,
            height,
            is_orbiting: false,
            is_panning: false,
            last_mouse_pos: None,
        };
        camera.update_matrices();
        camera
    }

    /// Set camera to view a bounding box
    pub fn look_at_bounds(&mut self, min: Vec3, max: Vec3) {
        self.target = (min + max) * 0.5;
        let size = max - min;
        let radius = size.length() * 0.5;
        self.distance = radius * 3.0; // Back off to see the whole volume
        self.update_matrices();
    }

    /// Get camera position in world space
    pub fn position(&self) -> Vec3 {
        self.target + Vec3::new(
            self.distance * self.yaw.cos() * self.pitch.cos(),
            self.distance * self.pitch.sin(),
            self.distance * self.yaw.sin() * self.pitch.cos(),
        )
    }

    /// Get view-projection matrix
    pub fn view_proj(&self) -> Mat4 {
        self.view_proj
    }

    /// Get inverse view-projection matrix
    pub fn inv_view_proj(&self) -> Mat4 {
        self.inv_view_proj
    }

    /// Handle window resize
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.width = width;
            self.height = height;
            self.update_matrices();
        }
    }

    /// Handle left mouse button press (start orbiting)
    pub fn start_orbit(&mut self) {
        self.is_orbiting = true;
        // Don't reset last_mouse_pos - let handle_mouse_move set it on first move
    }

    /// Handle right mouse button press (start panning)
    pub fn start_pan(&mut self) {
        self.is_panning = true;
        // Don't reset last_mouse_pos - let handle_mouse_move set it on first move
    }

    /// Handle mouse button release
    pub fn stop_drag(&mut self) {
        self.is_orbiting = false;
        self.is_panning = false;
        self.last_mouse_pos = None;  // Clear so next drag starts fresh
    }

    /// Handle mouse movement
    pub fn handle_mouse_move(&mut self, x: f32, y: f32) {
        if self.is_orbiting || self.is_panning {
            if let Some((last_x, last_y)) = self.last_mouse_pos {
                let dx = x - last_x;
                let dy = y - last_y;

                if self.is_orbiting {
                    self.orbit(dx, dy);
                } else if self.is_panning {
                    self.pan(dx, dy);
                }
            }
            // Always update last position when dragging
            self.last_mouse_pos = Some((x, y));
        }
    }

    /// Handle scroll wheel (zoom)
    pub fn handle_scroll(&mut self, delta: f32) {
        // Exponential zoom for smooth control at all scales
        let zoom_factor = 1.0 + delta * 0.1;
        self.distance /= zoom_factor;
        self.distance = self.distance.clamp(0.01, 100.0);
        self.update_matrices();
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.005;
        self.yaw -= dx * sensitivity;
        self.pitch += dy * sensitivity;

        // Clamp pitch to avoid gimbal lock
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.1;
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

        self.update_matrices();
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let eye = self.position();
        let forward = (self.target - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        // Scale pan speed with distance
        let pan_speed = self.distance * 0.002;
        self.target += right * (-dx * pan_speed) + up * (dy * pan_speed);

        self.update_matrices();
    }

    fn update_matrices(&mut self) {
        let eye = self.position();

        // View matrix
        self.view = Mat4::look_at_rh(eye, self.target, Vec3::Y);

        // Projection matrix
        let aspect = self.width as f32 / self.height as f32;
        self.proj = Mat4::perspective_rh(
            45.0_f32.to_radians(),
            aspect,
            0.001,
            100.0,
        );

        // Vulkan Y-flip correction
        let correction = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);

        self.view_proj = correction * self.proj * self.view;
        self.inv_view_proj = self.view_proj.inverse();
    }
}
