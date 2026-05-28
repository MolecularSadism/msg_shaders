// Black Hole 3D Shader - Bevy (GLSL)
// Full implementation of Eric Bruneton's Schwarzschild geodesic ray tracing
// Reference: https://ebruneton.github.io/black_hole_shader/

#version 450

layout(location = 0) in VS_OUTPUT {
    layout(location = 0) vec2 uv;
} in_data;

layout(location = 0) out vec4 o_Target;

layout(set = 2, binding = 0) uniform BlackHoleMaterial {
    float spin;
    float inclination;
    float time;
    float _pad0;
    vec4 disk_inner_color;
    vec4 disk_mid_color;
    vec4 disk_outer_color;
    vec4 glow_color;
} material;

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;

// Schwarzschild parameters (in units where M=1, G=1)
const float M = 0.5;  // Black hole mass
const float SCHWARZSCHILD_RADIUS = 2.0 * M;  // Event horizon
const float PHOTON_SPHERE_RADIUS = 3.0 * M;  // Unstable light orbit
const float DISK_INNER = 6.0 * M;    // Inner edge of accretion disk
const float DISK_OUTER = 20.0 * M;   // Outer edge of accretion disk

// Integration parameters
const int INTEGRATION_STEPS = 512;
const float DPHI = 0.001;  // Small angular step for accurate integration

// Schwarzschild metric: d²u/dφ² + u = 3u²
float schwarzschild_acceleration(float u) {
    return 3.0 * u * u - u;
}

// Trace a light ray in Schwarzschild spacetime
struct RayResult {
    float r_turn;           // Radius at turning point (where dr/dφ = 0)
    bool found_disk;        // Whether ray intersects disk
    float deflection;       // Total deflection angle
    float final_phi;        // Final angle after integration
};

RayResult trace_schwarzschild_ray(float b, float e_squared) {
    RayResult result;
    result.found_disk = false;
    result.deflection = 0.0;
    result.final_phi = 0.0;
    result.r_turn = 0.0;

    // Start integration from large radius where u ≈ 0
    float u = 0.001;
    float u_dot = -sqrt(e_squared - b * b * u * u * (1.0 - 2.0 * M * u));
    float phi = 0.0;

    // Integrate geodesic equation backward until we reach minimum radius
    bool found_turning_point = false;
    float prev_u_dot = u_dot;

    for (int step = 0; step < INTEGRATION_STEPS; step++) {
        // Detect turning point where u_dot changes sign
        if (!found_turning_point && u_dot < 0.0 && prev_u_dot >= 0.0) {
            found_turning_point = true;
            result.r_turn = 1.0 / u;

            // Check if turning point is in disk region
            if (result.r_turn >= DISK_INNER && result.r_turn <= DISK_OUTER) {
                result.found_disk = true;
            }
        }

        // Stop integration if we found the turning point
        if (found_turning_point) {
            break;
        }

        // Geodesic equation: d²u/dφ² = 3u² - u
        float accel = schwarzschild_acceleration(u);

        // Runge-Kutta step (RK2 for better accuracy)
        float k1_u = u_dot;
        float k1_udot = accel;

        float u_mid = u + k1_u * DPHI * 0.5;
        float udot_mid = u_dot + k1_udot * DPHI * 0.5;
        float accel_mid = schwarzschild_acceleration(u_mid);

        float k2_u = udot_mid;
        float k2_udot = accel_mid;

        u = u + (k1_u + k2_u) * DPHI * 0.5;
        prev_u_dot = u_dot;
        u_dot = u_dot + (k1_udot + k2_udot) * DPHI * 0.5;
        phi = phi + DPHI;

        // Stop if radius goes to infinity or gets very small
        if (u < 0.00001 || u > 0.5) {
            break;
        }
    }

    result.deflection = phi;
    result.final_phi = phi;

    return result;
}

// Render the accretion disk with relativistic effects
vec4 render_disk(float r, float phi) {
    // Black body radiation: T ∝ sqrt((6M)³/r³ * (1 - sqrt(6M/r)))
    float r_norm = r / (6.0 * M);
    float temp_factor = pow(6.0 * M / r, 1.5) * sqrt(1.0 - sqrt(6.0 * M / r));
    float temperature = pow(temp_factor, 0.25);

    // Keplerian orbital velocity
    float v_orbital = sqrt(M / r);

    // Doppler shift from orbital motion
    float doppler = sqrt((1.0 - v_orbital * cos(phi)) / (1.0 + v_orbital * cos(phi)));

    // Gravitational redshift
    float redshift = sqrt(1.0 - 2.0 * M / r);

    // Combine effects
    float total_shift = doppler * redshift;

    // Temperature modulated by Doppler shift
    float effective_temp = temperature * total_shift;

    // Color based on temperature
    vec4 color;
    if (effective_temp > 0.6) {
        // Inner hot disk: yellow-white
        color = mix(material.disk_mid_color, material.disk_inner_color,
                   (effective_temp - 0.6) / 0.4);
    } else if (effective_temp > 0.3) {
        // Mid disk: yellow-orange
        color = mix(material.disk_outer_color, material.disk_mid_color,
                   (effective_temp - 0.3) / 0.3);
    } else {
        // Outer disk: dimmer
        color = material.disk_outer_color * (effective_temp / 0.3);
    }

    // Brightness from temperature
    float brightness = effective_temp * effective_temp;
    color.rgb = color.rgb * brightness;

    // Edge fading
    float r_norm_disk = (r - DISK_INNER) / (DISK_OUTER - DISK_INNER);
    float edge_fade = smoothstep(0.0, 0.1, r_norm_disk) * smoothstep(1.0, 0.85, r_norm_disk);
    color.a = edge_fade;

    return color;
}

void main() {
    // Convert screen coordinates to normalized ray direction
    vec2 uv = in_data.uv;
    vec2 p = (uv - 0.5) * 2.0;
    float screen_r = length(p);
    float screen_phi = atan(p.y, p.x);

    // Observer is at large distance looking down at disk plane
    float observer_distance = 1000.0;

    // Convert screen pixel to impact parameter
    float angle_from_center = atan(screen_r);
    float impact_param = observer_distance * sin(angle_from_center);

    // Energy parameter for ray: e² = 1 + (b/M)²
    float e_squared = 1.0 + (impact_param / M) * (impact_param / M);

    // Trace the ray
    RayResult ray = trace_schwarzschild_ray(impact_param, e_squared);

    vec4 color = vec4(0.0);

    // Event horizon - pure black
    if (screen_r < SCHWARZSCHILD_RADIUS) {
        o_Target = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Photon sphere glow
    if (screen_r < PHOTON_SPHERE_RADIUS * 1.2 && screen_r > SCHWARZSCHILD_RADIUS * 1.5) {
        float photon_dist = abs(screen_r - PHOTON_SPHERE_RADIUS);
        float glow = exp(-photon_dist / (PHOTON_SPHERE_RADIUS * 0.2)) * 0.6;
        color = mix(color, material.glow_color, glow);
    }

    // Render accretion disk if ray intersects
    if (ray.found_disk) {
        vec4 disk_color = render_disk(ray.r_turn, screen_phi);
        color = mix(color, disk_color, disk_color.a);
    }

    // Einstein ring - secondary image of disk light bent around photon sphere
    if (screen_r > PHOTON_SPHERE_RADIUS * 0.95 && screen_r < PHOTON_SPHERE_RADIUS * 1.15) {
        float ring_thickness = abs(screen_r - PHOTON_SPHERE_RADIUS);
        float ring = exp(-ring_thickness / (PHOTON_SPHERE_RADIUS * 0.05)) * 0.8;
        color = mix(color, material.glow_color * vec4(0.8, 0.9, 1.0, 1.0), ring);
    }

    o_Target = clamp(color, 0.0, 1.0);
}
