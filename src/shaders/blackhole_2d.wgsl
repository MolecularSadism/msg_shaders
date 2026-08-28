// ============================================================================
// BLACK HOLE 2D SHADER - Schwarzschild Spacetime Visualization
// ============================================================================
// Simulates the visual appearance of a non-rotating black hole with:
// - Event Horizon: The Schwarzschild radius beyond which light cannot escape
// - Photon Sphere: At r = 1.5 Rs, where photons orbit the singularity
// - Accretion Disc: Orbiting plasma heated to incandescence by friction
// - Gravitational Lensing: Spacetime curvature bending light paths
// - Doppler Beaming: Relativistic brightening of approaching matter
// - Color Quantization: Optional retro-style palette reduction with dithering
//
// Based on Eric Bruneton's black_hole_shader (BSD-3-Clause)
// ============================================================================

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::color_quantize_functions as cq
#import msg_shaders::pixelate_functions as px

// ----------------------------------------------------------------------------
// UNIFORM BUFFER - Configurable parameters from Bevy
// ----------------------------------------------------------------------------
struct BlackHoleMaterial {
    // Row 1: Core dynamics
    spin: f32,              // Accretion disk angular velocity (rad/s)
    inclination: f32,       // Observer viewing angle (radians from disk plane)
    time: f32,              // Animation time for orbital motion
    shadow_radius: f32,     // Schwarzschild radius in UV space (event horizon)

    // Row 2: Disk geometry
    disk_inner_ratio: f32,  // ISCO radius as multiple of shadow_radius
    disk_outer_ratio: f32,  // Outer disk edge as multiple of shadow_radius
    photon_ring_width: f32, // Gaussian width of photon ring glow
    photon_ring_intensity: f32, // Peak brightness of photon ring

    // Row 3: Effects
    doppler_strength: f32,  // Doppler beaming intensity (0-1)
    cloud_density: f32,     // Accretion disk matter density
    axial_inner_ratio: f32,  // Secondary lensed disc axial inner boundary
    axial_outer_ratio: f32,  // Secondary lensed disc axial outer boundary

    // Row 4: Secondary ring parameters
    secondary_brightness: f32,          // Brightness multiplier for secondary rings
    outer_scale: f32,                   // Centered-UV span. Replaces the historical `* 2.0`.
    pixel_grid: f32,                    // Pixelation cells across the quad (0 = off).
    _pad2: f32,

    // Row 5-9: Emission colors by disk zone
    disk_inner_color: vec4<f32>,  // White-hot plasma near ISCO
    disk_mid_color: vec4<f32>,    // Intermediate temperature zone
    disk_outer_color: vec4<f32>,  // Cooler outer disk periphery
    glow_color: vec4<f32>,        // Photon ring emission
    black_color: vec4<f32>,       // Event horizon color
};

// Color quantization settings (optional - palette_size=0 disables)
struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    // Palette pre-converted to Oklab on the CPU so the shader skips per-pixel cbrt/pow.
    palette_oklab: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
};

@group(2) @binding(0)
var<uniform> material: BlackHoleMaterial;

@group(2) @binding(1)
var<uniform> quantization: QuantizationSettings;

// Baked nearest-palette LUT (Rgba32Float). Point-loaded (no sampler) to replace
// the per-pixel Oklab palette loop with one fetch at each quantization site.
@group(2) @binding(2)
var palette_lut: texture_3d<f32>;

// ----------------------------------------------------------------------------
// MATHEMATICAL CONSTANTS
// ----------------------------------------------------------------------------
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;  // 2π - Full orbital period

// ----------------------------------------------------------------------------
// PROCEDURAL NOISE FUNCTIONS
// Used to generate turbulent structure in the accretion disk
// ----------------------------------------------------------------------------

/// Pseudo-random hash function for deterministic noise generation.
/// Maps any float to a seemingly random value in [0, 1).
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

/// 2D hash for spatially-varying noise patterns.
/// Creates unique values based on 2D position for disk turbulence.
fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

/// Smooth 2D value noise with cubic interpolation.
/// Generates organic-looking patterns for accretion disk structure.
fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Hermite interpolation for smooth gradients
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(hash2(i + vec2<f32>(0.0, 0.0)), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

// ----------------------------------------------------------------------------
// ACCRETION DISK CLOUD LAYER
// Generates a single layer of orbiting matter clumps
// ----------------------------------------------------------------------------

/// Cloud layer configuration parameters
struct CloudLayerParams {
    angular_count: f32,    // Number of clouds around the orbit
    radial_count: f32,     // Number of radial bands
    phase_offset: f32,     // Angular offset for layer variety
    radial_offset: f32,    // Radial phase offset
    visibility_threshold: f32,  // Particle density threshold
    angular_falloff: f32,  // How quickly clouds fade angularly
    radial_falloff: f32,   // How quickly clouds fade radially
    brightness: f32,       // Layer contribution weight
    chaos_seed: f32,       // Unique seed for this layer's chaos
}

/// Generates a single cloud layer's contribution to disk brightness.
/// Models discrete matter clumps orbiting in Keplerian motion.
///
/// `band12` is `floor(norm_r * 12.0)`, a coarse radial band index that is
/// independent of layer params and therefore hoisted to the caller so the
/// seven layers share one `floor` instead of computing seven.
fn cloud_layer(
    phi: f32,           // Azimuthal angle in disk plane
    norm_r: f32,        // Normalized radius [0,1] within disk
    rotating_phi: f32,  // Time-animated angular position
    band12: f32,        // Shared coarse band index `floor(norm_r * 12.0)`
    params: CloudLayerParams
) -> f32 {
    // Per-band angular chaos creates realistic turbulent structure
    let radial_chaos = hash(floor(norm_r * params.chaos_seed) * 73.1) * TAU;
    let chaotic_phi = rotating_phi + radial_chaos + params.phase_offset;

    // Integer count: clouds tile the ring seamlessly and the layer repeats each revolution.
    let ang_count = max(round(params.angular_count + hash(band12 * params.chaos_seed) * 20.0), 1.0);

    // Convert to grid coordinates for discrete cloud placement
    let ang_phase = chaotic_phi * ang_count / TAU;
    let rad_phase = norm_r * params.radial_count + params.radial_offset;

    let ang_id = floor(ang_phase);
    let rad_id = floor(rad_phase);
    let ang_frac = fract(ang_phase);
    let rad_frac = fract(rad_phase);

    // Slot wrapped into [0, ang_count) so cloud identities repeat each revolution and never grow unbounded.
    let ang_slot = ang_id - ang_count * floor(ang_id / ang_count);
    let cloud_id = ang_slot * 7.0 + rad_id * 13.0 + hash(rad_id * 97.0) * 100.0;

    // Stochastic visibility based on density parameter
    let density_adjusted_threshold = params.visibility_threshold / material.cloud_density;
    let cloud_visible = step(density_adjusted_threshold, hash(cloud_id * 17.3));

    // Cloud center offset for organic irregularity
    let center_offset = hash(cloud_id * 51.7) * 0.3 - 0.15;
    let ang_dist = abs(ang_frac - 0.5 + center_offset);
    let rad_dist = abs(rad_frac - 0.5);

    // Per-cloud width variation
    let width_var = 0.7 + hash(cloud_id * 67.3) * 0.6;

    // Gaussian falloff for soft cloud edges
    let ang_falloff = exp(-ang_dist * ang_dist * params.angular_falloff * width_var);
    let rad_falloff = exp(-rad_dist * rad_dist * params.radial_falloff);

    return ang_falloff * rad_falloff * cloud_visible * params.brightness;
}

// ----------------------------------------------------------------------------
// ACCRETION DISK STRUCTURE
// Combines multiple cloud layers into complete disk appearance
// ----------------------------------------------------------------------------

/// Generates the orbital cloud pattern across the entire accretion disk.
/// Returns brightness value [0,1] representing matter density at this point.
fn accretion_disk_clouds(phi: f32, r: f32, disk_inner: f32, disk_outer: f32) -> f32 {
    // Keplerian orbital motion - inner disk rotates faster
    let rotating_phi = phi + material.time * material.spin * 0.5;

    // Normalize radius to [0,1] within disk bounds
    let norm_r = (r - disk_inner) / (disk_outer - disk_inner);

    // Soft edge fade at disk boundaries (prevents hard cutoff)
    let edge_fade = smoothstep(0.0, 0.12, norm_r) * smoothstep(1.0, 0.88, norm_r);

    // Coarse radial band index shared by every layer's ang_count hash.
    let band12 = floor(norm_r * 12.0);

    var brightness = 0.0;

    // Layer 1: Primary large-scale structure (spiral arms)
    brightness = max(brightness, cloud_layer(phi, norm_r, rotating_phi, band12, CloudLayerParams(
        40.0, 44.0, 0.0, 0.0, 0.22, 6.45, 6.0, 1.0, 20.0
    )));

    // Layer 2: Secondary turbulent eddies
    brightness = max(brightness, cloud_layer(phi, norm_r, rotating_phi, band12, CloudLayerParams(
        55.0, 60.0, 0.5, 0.33, 0.25, 7.85, 8.0, 0.8, 15.0
    )));

    // Layer 3: Fine-scale turbulence
    brightness = max(brightness, cloud_layer(phi, norm_r, rotating_phi, band12, CloudLayerParams(
        80.0, 90.0, 0.0, 0.67, 0.27, 10.0, 10.0, 0.6, 25.0
    )));

    // Layer 4: Wispy filaments
    brightness = max(brightness, cloud_layer(phi, norm_r, rotating_phi, band12, CloudLayerParams(
        120.0, 120.0, 0.25, 0.0, 0.30, 12.5, 12.5, 0.4, 30.0
    )));

    // Layer 5: Additional density variations
    brightness = max(brightness, cloud_layer(phi, norm_r, rotating_phi, band12, CloudLayerParams(
        65.0, 52.0, 0.75, 0.17, 0.24, 8.5, 7.0, 0.9, 17.0
    )));

    // Layer 6: Intermediate structure
    brightness = max(brightness, cloud_layer(phi, norm_r, rotating_phi, band12, CloudLayerParams(
        95.0, 75.0, 0.125, 0.5, 0.28, 9.5, 9.0, 0.7, 21.0
    )));

    // Layer 7: Largest-scale density waves
    brightness = max(brightness, cloud_layer(phi, norm_r, rotating_phi, band12, CloudLayerParams(
        48.0, 38.0, 0.375, 0.83, 0.26, 5.5, 5.5, 0.85, 23.0
    )));

    return clamp(brightness * edge_fade, 0.0, 1.0);
}

// ----------------------------------------------------------------------------
// RELATIVISTIC DOPPLER EFFECTS
// Models the brightening/dimming of matter moving toward/away from observer
// ----------------------------------------------------------------------------

/// Calculates Doppler beaming factor for orbiting matter.
/// Matter approaching the observer (left side for CCW rotation) appears brighter
/// due to relativistic aberration and blueshift.
fn doppler_beaming(phi: f32) -> f32 {
    // phi = PI corresponds to approaching side (left), phi = 0 to receding (right)
    let base = 0.35 + 0.65 * (1.0 - cos(phi)) * 0.5;
    return mix(1.0, base, material.doppler_strength);
}

/// Horizontal Doppler effect for screen-space relativistic dimming.
/// Right side of disk (receding matter) shows redshift and reduced brightness.
fn horizontal_doppler(x: f32) -> vec4<f32> {
    // Shift effect onset 0.1 earlier (into negative x territory)
    let t = clamp(x + 0.11, 0.0, 1.0);
    let strength = material.doppler_strength;

    // Transparency falloff - delayed onset so color darkens first
    let t_alpha = clamp((t - 0.2) / 0.85, 0.0, 1.0);
    let f = 1.0 - t_alpha * strength;
    let f2 = f * f;
    let f4 = f2 * f2;
    let falloff = f4 * f2; // f^6

    // Differential redshift: blue fades fastest, red persists longest.
    // Mul chains (vs `pow(x, c)`) bypass the `exp(c * log(x))` lowering that
    // naga emits for the pow builtin; with integer exponents this is the
    // exact same result, with far less work.
    let s = 1.0 - t * strength;
    let s2 = s * s;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    let r_factor = s4;                  // s^4
    let g_factor = s8 * s2;             // s^10
    let b_factor = s8 * s4 * s2;        // s^14

    return vec4<f32>(r_factor, g_factor, b_factor, falloff);
}

// ----------------------------------------------------------------------------
// DISK COLOR AND EMISSION
// Temperature-based coloring following black body radiation principles
// ----------------------------------------------------------------------------

/// Calculates disk emission color and transparency at given disk coordinates.
/// Inner regions are hotter (whiter), outer regions cooler (redder).
/// Edge fading is applied here so it's baked into the disk before compositing.
fn disk_emission(r: f32, phi: f32, disk_inner: f32, disk_outer: f32) -> vec4<f32> {
    let t = (r - disk_inner) / (disk_outer - disk_inner);

    // Edge fades - transparent near inner and outer boundaries
    let inner_fade = smoothstep(0.0, 0.15, t);
    let outer_fade = smoothstep(1.0, 0.85, t);

    // Temperature gradient colors (approximating black body radiation)
    let white_hot = vec3<f32>(1.0, 0.98, 0.9);   // ~10,000K plasma
    let bright_yellow = vec3<f32>(1.0, 0.95, 0.3); // ~6,000K

    var col: vec3<f32>;

    if t < 0.25 {
        // Corona: Innermost white-hot region near ISCO
        let inner_t = t / 0.25;
        col = mix(white_hot, mix(white_hot, bright_yellow, 0.5), inner_t);
    } else if t < 0.55 {
        // Hot zone: Bright yellow emission
        let yellow_t = (t - 0.25) / 0.3;
        let yellow_white = mix(white_hot, bright_yellow, 0.5);
        col = mix(yellow_white, bright_yellow, yellow_t);
    } else if t < 0.85 {
        // Transition zone: Yellow to mid-disk color
        let mid_t = (t - 0.55) / 0.3;
        col = mix(bright_yellow, material.disk_mid_color.rgb, mid_t);
    } else {
        // Outer disk: Cooling material with redshift (last 15%)
        let outer_t = (t - 0.85) / 0.15;
        let base_col = mix(material.disk_mid_color.rgb, material.disk_outer_color.rgb, outer_t);
        let red_shift = outer_t * outer_t;
        let darkness = 1.0 - red_shift * 0.5;
        col = mix(base_col, vec3<f32>(base_col.r * 0.8, base_col.g * 0.25, base_col.b * 0.1), red_shift) * darkness;
    }

    // Cloud pattern determines both brightness and opacity
    let cloud = accretion_disk_clouds(phi, r, disk_inner, disk_outer);

    // Angle-dependent density cutoff (denser on approaching side)
    let angle_factor = -(1.0 + cos(phi)) * 0.5;
    let cutoff = mix(0.03, 0.35, angle_factor);
    let cloud_cut = max(cloud - cutoff, 0.0) / (1.0 - cutoff);

    // No emission without clouds - eliminate base brightness
    if cloud_cut < 0.05 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Fine texture variation
    let rot_phi = phi + material.time * material.spin;
    let variation = 0.95 + 0.05 * sin(rot_phi * 8.0);

    // Radial brightness falloff (clamped to prevent excessive brightness)
    let radial_brightness = min(1.0 / (0.15 + t * 0.1), 4.0);

    // Cloud density affects color intensity (sparse = dimmer)
    // Edge fades affect alpha (boundaries = transparent)
    return vec4<f32>(col * radial_brightness * cloud_cut * variation, inner_fade * outer_fade);
}


// ----------------------------------------------------------------------------
// MAIN FRAGMENT SHADER
// Composites all visual elements: accretion disc, shadow, photon ring, lensing
// ----------------------------------------------------------------------------

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Optional pixelation: snap the quad UV to a grid so the whole effect
    // renders as discrete blocks. A grid of 0 passes the UV through unchanged.
    let src_uv = px::pixelate_uv(in.uv, vec2<f32>(material.pixel_grid, material.pixel_grid));

    // Convert UV to centered coordinates. The span (`outer_scale`) is set by
    // the CPU so that the visible disc edge lands at the same internal
    // coordinate as before, even when the quad has been shrunk to fit the
    // disc. The shader's geometry constants are unchanged.
    let uv = (src_uv - 0.5) * material.outer_scale;
    let x = uv.x;
    let y = uv.y;

    // Screen-space polar coordinates
    let screen_r = length(uv);
    let screen_phi = atan2(y, x);

    // Calculate disk geometry from uniform parameters
    let shadow_r = material.shadow_radius;
    let disk_inner = shadow_r * material.disk_inner_ratio;
    let disk_outer = shadow_r * material.disk_outer_ratio;

    // Tight early-exit: nothing visible past the disc outer edge plus a small
    // margin for the Gaussian falloff in `disk_emission`. Discards quad-corner
    // pixels that the disc never reaches.
    if screen_r > disk_outer * 1.02 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Observer inclination (viewing angle above disk plane)
    let inc = material.inclination;
    let cos_inc = cos(inc);
    let sin_inc = sin(inc);

    var final_color = vec3<f32>(0.0);
    var final_alpha = 0.0;

    // ========================================================================
    // ACCRETION DISC GEOMETRY
    // ========================================================================
    // The disc lies in the x-z plane (y=0 in 3D space).
    // Camera views from inclination angle above the disc.
    // Screen y is compressed by cos(inclination) due to perspective.

    let disk_x = -x;
    let disk_z = -y / max(abs(cos_inc), 0.001);
    let disk_r = length(vec2<f32>(disk_x, disk_z));
    let disk_phi = atan2(disk_z, disk_x);

    // Accretion disc/lensed disc transition (smooth blend near disc plane)
    let transition_width = 0.1;
    let front_factor = smoothstep(-transition_width, transition_width, -disk_z);

    // ========================================================================
    // ACCRETION DISC (Direct View) — fires only for y > 0
    // ========================================================================
    // Hoisted above the shadow check because the shadow path uses disc_color
    // as the overlay inside the event horizon shadow region.
    var disc_color = vec3<f32>(0.0);
    var disc_alpha = 0.0;

    if y > 0.0 && disk_r >= disk_inner && disk_r <= disk_outer {
        let d = doppler_beaming(disk_phi);
        let disk_data = disk_emission(disk_r, disk_phi, disk_inner, disk_outer);
        disc_color = disk_data.rgb * d;
        disc_alpha = disk_data.a;
    }

    // Secondary-layer state (computed below for non-shadow pixels only).
    var lensed_color = vec3<f32>(0.0);
    var lensed_alpha = 0.0;
    var axial_color = vec3<f32>(0.0);
    var axial_alpha = 0.0;
    var equatorial_color = vec3<f32>(0.0);
    var equatorial_alpha = 0.0;

    let axial_inner = shadow_r * material.axial_inner_ratio;
    let axial_outer = shadow_r * material.axial_outer_ratio;

    // For shadow-region pixels, the secondary layers are never composited
    // (the shadow branch below returns before alpha compositing). Computing
    // them only outside the shadow saves up to two `disk_emission` calls
    // (and 14 cloud-layer evaluations) per shadow pixel.
    if screen_r >= shadow_r {
        // y-sign dispatch: lensed/equatorial fire only for y < 0, axial only
        // for y > 0. Splitting by sign skips half of the condition checks.
        if y > 0.0 {
            // ----------------------------------------------------------------
            // SECONDARY LENSED DISC AXIAL (thin ring above the shadow)
            // ----------------------------------------------------------------
            if screen_r >= axial_inner && screen_r <= axial_outer {
                let axial_t = (screen_r - axial_inner) / (axial_outer - axial_inner);
                let source_r = mix(disk_inner, disk_outer, axial_t);
                let source_phi = screen_phi;

                let d = doppler_beaming(source_phi);
                let disk_data = disk_emission(source_r, source_phi, disk_inner, disk_outer);

                axial_color = disk_data.rgb * d * material.secondary_brightness;
                axial_alpha = disk_data.a;
            }
        } else {
            // ----------------------------------------------------------------
            // PRIMARY LENSED DISC (Gravitationally Lensed View)
            // ----------------------------------------------------------------
            // Forms a circular annulus around the shadow. Near the disc
            // plane (y≈0) it expands to meet the accretion disc; far from
            // it, it hugs the event horizon more closely.
            let lensed_inner_base = shadow_r * 1.08;
            let lensed_outer_base = shadow_r * 2.2;
            let y_factor = clamp(abs(y) / lensed_outer_base, 0.0, 1.0);
            let bend_strength = exp(-y_factor * y_factor * 6.0);
            let lensed_inner = mix(lensed_inner_base, disk_inner, bend_strength);
            let expanded_outer = mix(lensed_outer_base, disk_outer, bend_strength);

            if screen_r >= lensed_inner && screen_r <= expanded_outer {
                let lensed_t = (screen_r - lensed_inner) / (expanded_outer - lensed_inner);
                let source_r = mix(disk_inner, disk_outer, lensed_t);
                let source_phi = screen_phi + PI;

                let d = doppler_beaming(source_phi);
                let disk_data = disk_emission(source_r, source_phi, disk_inner, disk_outer);

                lensed_color = disk_data.rgb * d * 0.9;
                lensed_alpha = disk_data.a;
            }

            // ----------------------------------------------------------------
            // SECONDARY LENSED DISC EQUATORIAL (behind the shadow, elliptical)
            // ----------------------------------------------------------------
            // Wraps BEHIND the black hole, elliptical due to inclination.
            // Fires only when disk_z > 0, which under positive cos(inc) is
            // equivalent to y < 0.
            let eq_inner_base = shadow_r * 1.01;
            let eq_outer_base = shadow_r * 1.2;
            let eq_bend = exp(-abs(disk_z) * abs(disk_z) * 2.0);
            let eq_inner_expanded = mix(eq_inner_base, axial_inner, eq_bend);
            let eq_outer_expanded = mix(eq_outer_base, axial_outer, eq_bend);

            if disk_z > 0.0 && disk_r >= eq_inner_expanded && disk_r <= eq_outer_expanded {
                let equatorial_t = (disk_r - eq_inner_expanded) / (eq_outer_expanded - eq_inner_expanded);
                let source_r = mix(disk_inner, disk_outer, equatorial_t);
                let source_phi = disk_phi;

                let d = doppler_beaming(source_phi);
                let disk_data = disk_emission(source_r, source_phi, disk_inner, disk_outer);

                equatorial_color = disk_data.rgb * d * material.secondary_brightness;
                equatorial_alpha = disk_data.a;
            }
        }
    }

    // ========================================================================
    // EVENT HORIZON (Schwarzschild Shadow)
    // ========================================================================
    // The event horizon is rendered as the absolute black region from which
    // no light can escape. The accretion disc may overlay the shadow.
    //
    // When quantization is enabled, disc and shadow are quantized separately.
    // The disc's alpha determines whether we see quantized disc or quantized black.
    // This allows transparent disc regions to show pure black through dithering.

    if screen_r < shadow_r {
        let screen_pos = in.position.xy;

        if quantization.palette_size > 0u {
            // QUANTIZATION PATH: Quantize each disc layer SEPARATELY - NO pre-blending at all
            // Each layer quantizes independently, then composite quantized results

            // Pre-multiply Doppler so we can test transparency without the
            // full palette search. `quantize_color`'s `effective_alpha` is
            // bounded above by the input alpha (the luminance-weighted mix
            // only reduces it), so an input alpha below `alpha_cutoff`
            // guarantees a transparent quantized result. In that case the
            // 64-entry nearest-neighbor search is wasted work — the only
            // thing we need is the quantized black below.
            var quantized_disc = vec4<f32>(0.0);
            if front_factor > 0.5 {
                let h_doppler_main = horizontal_doppler(x);
                let disc_input_alpha = disc_alpha * h_doppler_main.a;
                if disc_input_alpha >= quantization.alpha_cutoff {
                    let main_with_doppler = vec4<f32>(disc_color * h_doppler_main.rgb, disc_input_alpha);
                    quantized_disc = cq::quantize_color_lut(
                        main_with_doppler,
                        screen_pos,
                        palette_lut,
                        quantization.palette_size,
                        quantization.alpha_cutoff,
                        quantization.transparency_floor,
                        quantization.dither_pattern
                    );
                }
            }

            // If disc is present (alpha above cutoff), show disc. Otherwise show black.
            // No mixing - disc continues as if shadow wasn't there.
            if quantized_disc.a <= quantization.alpha_cutoff {
                let quantized_black = cq::quantize_color_lut(
                    vec4<f32>(material.black_color.rgb, 1.0),
                    screen_pos,
                    palette_lut,
                    quantization.palette_size,
                    quantization.alpha_cutoff,
                    quantization.transparency_floor,
                    quantization.dither_pattern
                );
                return vec4<f32>(quantized_black.rgb, 1.0);
            }
        } else {
            // NO QUANTIZATION PATH: Standard blending
            var overlay_color = lensed_color;
            var overlay_alpha = lensed_alpha;

            // Add main disc if it's in front
            if front_factor > 0.5 {
                overlay_color = disc_color + overlay_color * (1.0 - disc_alpha);
                overlay_alpha = disc_alpha + overlay_alpha * (1.0 - disc_alpha);
            }

            // Apply horizontal Doppler effect
            let h_doppler = horizontal_doppler(x);
            overlay_color = overlay_color * h_doppler.rgb;
            overlay_alpha = overlay_alpha * h_doppler.a;

            // Standard alpha blend over black
            let blended = overlay_color + material.black_color.rgb * (1.0 - overlay_alpha);
            return vec4<f32>(blended, 1.0);
        }
    }

    // ========================================================================
    // ALPHA COMPOSITING (All disc layers)
    // ========================================================================
    // Compositing order (back to front):
    // 1. Secondary lensed disc equatorial (furthest back, below y=0)
    // 2. Secondary lensed disc axial (thin ring at top, behind accretion disc)
    // 3. Accretion disc (primary, above y=0)
    // 4. Primary lensed disc (main bottom arc, frontmost)
    // C_out = C_front + C_back * (1 - A_front)
    // A_out = A_front + A_back * (1 - A_front)

    // Start with secondary lensed disc equatorial (backmost layer)
    final_color = equatorial_color;
    final_alpha = equatorial_alpha;

    // Composite secondary lensed disc axial over equatorial
    final_color = axial_color + final_color * (1.0 - axial_alpha);
    final_alpha = axial_alpha + final_alpha * (1.0 - axial_alpha);

    // Composite accretion disc over axial
    final_color = disc_color + final_color * (1.0 - disc_alpha);
    final_alpha = disc_alpha + final_alpha * (1.0 - disc_alpha);

    // Composite primary lensed disc on top (frontmost)
    final_color = lensed_color + final_color * (1.0 - lensed_alpha);
    final_alpha = lensed_alpha + final_alpha * (1.0 - lensed_alpha);

    // ========================================================================
    // PHOTON RING (Light at the Photon Sphere)
    // ========================================================================
    // At r = 1.5 Rs, photons orbit the black hole in unstable circular orbits.
    // This creates a bright ring just outside the event horizon.

    let ring_r = shadow_r * 1.01;
    let ring_width = material.photon_ring_width;
    let ring_dist = abs(screen_r - ring_r);
    let ring_intensity = exp(-ring_dist * ring_dist / (ring_width * ring_width)) * material.photon_ring_intensity;

    if ring_intensity > 0.05 {
        let d = doppler_beaming(screen_phi);
        final_color = max(final_color, material.glow_color.rgb * ring_intensity * d);
        final_alpha = max(final_alpha, ring_intensity);
    }

    // ========================================================================
    // HORIZONTAL DOPPLER (Screen-space Relativistic Effect)
    // ========================================================================
    let h_doppler = horizontal_doppler(x);
    final_color = final_color * h_doppler.rgb;
    final_alpha = final_alpha * h_doppler.a;

    let output = vec4<f32>(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(final_alpha, 0.0, 1.0));

    // ========================================================================
    // COLOR QUANTIZATION (Optional Retro-style Palette Reduction)
    // ========================================================================
    // When palette_size > 0, apply quantization with dithering.
    // Note: shadow area (screen_r < shadow_r) is handled earlier with early return.
    if quantization.palette_size > 0u {
        let screen_pos = in.position.xy;

        return cq::quantize_color_lut(
            output,
            screen_pos,
            palette_lut,
            quantization.palette_size,
            quantization.alpha_cutoff,
            quantization.transparency_floor,
            quantization.dither_pattern
        );
    }

    return output;
}
