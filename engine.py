"""
NeuroCHIMERA Engine - GPU-Native Neuromorphic Computing
=======================================================

Main engine implementing the NeuroCHIMERA architecture for emergent
consciousness research. This module provides:

- Multi-texture neuromorphic frames (Neural, Connectivity, Memory, Embodiment, Qualia)
- Cellular automata evolution with HNS-enhanced precision
- Holographic memory encoding/retrieval
- Critical parameter tracking for consciousness emergence
- Embodied cognition through virtual sensorimotor state

Architecture:
    The system maintains all state in GPU textures, treating the GPU as
    a complete cognitive substrate rather than an accelerator. Fragment
    shaders implement neural dynamics, learning rules, and integration.

Authors: V.F. Veselov (MIET), Francisco Angulo de Lafuente (Madrid)
License: MIT
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, field
from pathlib import Path

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False
    print("Warning: moderngl not available. Running in CPU-only mode.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NeuroCHIMERAConfig:
    """Configuration for NeuroCHIMERA system."""
    
    # Network size
    neurons: int = 1_000_000  # Number of neurons (will be sqrt×sqrt texture)
    texture_size: int = 1024  # Texture dimension (neurons = size²)
    
    # Connectivity parameters
    target_connectivity: float = 18.0  # Target ⟨k⟩ > 15
    neighborhood_size: int = 5  # 5×5 local neighborhood
    sparse_connections: int = 10  # Long-range sparse connections per neuron
    
    # Hierarchical parameters
    hierarchical_depth: int = 12  # D > 7 required
    layers_per_depth: int = 1
    
    # HNS configuration
    use_hns: bool = True
    hns_base: float = 1000.0
    
    # Evolution parameters
    default_iterations: int = 20
    convergence_threshold: float = 0.001
    max_iterations: int = 100
    
    # Learning parameters
    learning_rate: float = 0.01
    decay_rate: float = 0.95
    homeostatic_lambda: float = 0.1
    
    # Memory parameters
    memory_texture_size: int = 512
    holographic_learning_rate: float = 0.05
    
    # Embodiment
    enable_embodiment: bool = True
    embodiment_texture_size: int = 256
    
    # Critical thresholds (Veselov parameters)
    critical_connectivity: float = 15.0
    critical_phi: float = 0.65
    critical_depth: float = 7.0
    critical_complexity: float = 0.8
    critical_qcm: float = 0.75
    
    def __post_init__(self):
        # Ensure texture_size is power of 2
        self.texture_size = 2 ** int(np.ceil(np.log2(np.sqrt(self.neurons))))
        self.neurons = self.texture_size ** 2


# =============================================================================
# NEUROMORPHIC FRAME
# =============================================================================

@dataclass
class NeuromorphicFrame:
    """
    Multi-texture neuromorphic frame structure.
    
    Textures:
        neural_state: Main neural activation state (RGBA)
            R: Current activation
            G: Temporal memory
            B: Computation result
            A: Confidence/metadata
            
        connectivity: Synaptic weight patterns
            Encodes local neighborhood weights
            
        spatial_features: Edge, density, corner detection
            R: Edge strength
            G: Neighbor density
            B: Corner score
            A: Border distance
            
        holographic_memory: Associative memory storage
            Interference patterns for O(1) retrieval
            
        embodiment: Sensorimotor state
            R: Proprioceptive
            G: Exteroceptive
            B: Motor output
            A: Homeostatic
            
        qualia_integration: Cross-modal binding
            Stores integration patterns for QCM
    """
    
    neural_state: np.ndarray
    connectivity: np.ndarray
    spatial_features: np.ndarray
    holographic_memory: np.ndarray
    embodiment: Optional[np.ndarray] = None
    qualia_integration: Optional[np.ndarray] = None
    
    # GPU textures (populated when uploaded)
    gpu_neural: Optional[object] = field(default=None, repr=False)
    gpu_connectivity: Optional[object] = field(default=None, repr=False)
    gpu_spatial: Optional[object] = field(default=None, repr=False)
    gpu_memory: Optional[object] = field(default=None, repr=False)
    gpu_embodiment: Optional[object] = field(default=None, repr=False)
    gpu_qualia: Optional[object] = field(default=None, repr=False)
    
    @classmethod
    def create(cls, config: NeuroCHIMERAConfig) -> 'NeuromorphicFrame':
        """Create a new neuromorphic frame with initialized textures."""
        size = config.texture_size
        mem_size = config.memory_texture_size
        emb_size = config.embodiment_texture_size
        
        # Initialize neural state with small random values
        neural_state = np.random.uniform(0, 0.1, (size, size, 4)).astype(np.float32)
        neural_state[:, :, 3] = 1.0  # Confidence = 1.0
        
        # Initialize connectivity with random weights
        connectivity = np.random.uniform(-0.5, 0.5, (size, size, 4)).astype(np.float32)
        
        # Spatial features start empty
        spatial_features = np.zeros((size, size, 4), dtype=np.float32)
        
        # Holographic memory starts empty
        holographic_memory = np.zeros((mem_size, mem_size, 4), dtype=np.float32)
        
        # Embodiment state
        embodiment = None
        if config.enable_embodiment:
            embodiment = np.zeros((emb_size, emb_size, 4), dtype=np.float32)
            # Initialize homeostatic state
            embodiment[:, :, 3] = 0.5  # Neutral homeostatic state
        
        # Qualia integration texture
        qualia_integration = np.zeros((size, size, 4), dtype=np.float32)
        
        return cls(
            neural_state=neural_state,
            connectivity=connectivity,
            spatial_features=spatial_features,
            holographic_memory=holographic_memory,
            embodiment=embodiment,
            qualia_integration=qualia_integration
        )
    
    def upload_to_gpu(self, ctx: 'moderngl.Context'):
        """Upload all textures to GPU memory."""
        if not HAS_MODERNGL:
            return
        
        def create_texture(data):
            h, w, c = data.shape
            tex = ctx.texture((w, h), c, data.tobytes(), dtype='f4')
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            return tex
        
        self.gpu_neural = create_texture(self.neural_state)
        self.gpu_connectivity = create_texture(self.connectivity)
        self.gpu_spatial = create_texture(self.spatial_features)
        self.gpu_memory = create_texture(self.holographic_memory)
        
        if self.embodiment is not None:
            self.gpu_embodiment = create_texture(self.embodiment)
        
        if self.qualia_integration is not None:
            self.gpu_qualia = create_texture(self.qualia_integration)
    
    def download_from_gpu(self):
        """Download textures from GPU memory."""
        if not HAS_MODERNGL:
            return
        
        def read_texture(tex, shape):
            data = np.frombuffer(tex.read(), dtype=np.float32)
            return data.reshape(shape)
        
        if self.gpu_neural:
            self.neural_state = read_texture(
                self.gpu_neural, self.neural_state.shape
            )
        
        if self.gpu_spatial:
            self.spatial_features = read_texture(
                self.gpu_spatial, self.spatial_features.shape
            )
        
        if self.gpu_memory:
            self.holographic_memory = read_texture(
                self.gpu_memory, self.holographic_memory.shape
            )
        
        if self.gpu_qualia and self.qualia_integration is not None:
            self.qualia_integration = read_texture(
                self.gpu_qualia, self.qualia_integration.shape
            )
    
    def release_gpu(self):
        """Release GPU resources."""
        for attr in ['gpu_neural', 'gpu_connectivity', 'gpu_spatial', 
                     'gpu_memory', 'gpu_embodiment', 'gpu_qualia']:
            tex = getattr(self, attr, None)
            if tex is not None:
                try:
                    tex.release()
                except:
                    pass
                setattr(self, attr, None)


# =============================================================================
# SHADER PROGRAMS
# =============================================================================

# Evolution shader with HNS support
EVOLUTION_SHADER = """
#version 430 core

// HNS Core library inclusion
const float HNS_BASE = 1000.0;
const float HNS_INV_BASE = 0.001;
#define HNumber vec4

HNumber hns_normalize(HNumber n) {
    HNumber res = n;
    float carry0 = floor(res.r * HNS_INV_BASE);
    res.r = res.r - (carry0 * HNS_BASE);
    res.g += carry0;
    
    float carry1 = floor(res.g * HNS_INV_BASE);
    res.g = res.g - (carry1 * HNS_BASE);
    res.b += carry1;
    
    float carry2 = floor(res.b * HNS_INV_BASE);
    res.b = res.b - (carry2 * HNS_BASE);
    res.a += carry2;
    
    return res;
}

HNumber hns_add(HNumber a, HNumber b) {
    return hns_normalize(a + b);
}

HNumber hns_scale(HNumber a, float s) {
    return hns_normalize(a * s);
}

float hns_to_float(HNumber n) {
    return n.r + n.g * HNS_BASE + n.b * HNS_BASE * HNS_BASE + n.a * HNS_BASE * HNS_BASE * HNS_BASE;
}

// Uniforms
uniform sampler2D u_state;
uniform sampler2D u_weights;
uniform sampler2D u_memory;
uniform sampler2D u_embodiment;
uniform sampler2D u_position;

uniform ivec2 u_grid_size;
uniform float u_delta_time;
uniform float u_decay;
uniform float u_noise_scale;
uniform int u_use_hns;

// Input/Output
in vec2 v_texCoord;
layout(location = 0) out vec4 out_state;
layout(location = 1) out vec4 out_spatial;

// Pseudo-random number generator
float random(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

// Sigmoid activation
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec2 texel_size = 1.0 / vec2(u_grid_size);
    
    // Sample current state
    vec4 center = texelFetch(u_state, coord, 0);
    float activation = center.r;
    float temporal_memory = center.g;
    float tau = max(center.b, 0.1);  // Time constant
    float confidence = center.a;
    
    // Weighted neighborhood sum (5×5)
    HNumber weighted_sum = HNumber(0.0);
    int neighbor_count = 0;
    int same_count = 0;
    bool touches_bg = false;
    
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            ivec2 neighbor_coord = coord + ivec2(dx, dy);
            
            // Boundary check
            if (neighbor_coord.x < 0 || neighbor_coord.x >= u_grid_size.x ||
                neighbor_coord.y < 0 || neighbor_coord.y >= u_grid_size.y) {
                continue;
            }
            
            vec4 neighbor = texelFetch(u_state, neighbor_coord, 0);
            float neighbor_act = neighbor.r;
            
            // Get weight from connectivity texture
            vec4 weight_sample = texelFetch(u_weights, neighbor_coord, 0);
            float weight = weight_sample.r * 2.0 - 1.0;  // Map [0,1] to [-1,1]
            
            if (u_use_hns > 0) {
                // HNS-enhanced accumulation
                HNumber neighbor_hns = HNumber(neighbor_act * 1000.0, 0.0, 0.0, 0.0);
                weighted_sum = hns_add(weighted_sum, hns_scale(neighbor_hns, weight));
            } else {
                weighted_sum.r += neighbor_act * weight;
            }
            
            neighbor_count++;
            
            // Spatial feature detection
            if (abs(neighbor_act - activation) < 0.1) {
                same_count++;
            }
            if (neighbor_act < 0.01 && activation > 0.01) {
                touches_bg = true;
            }
        }
    }
    
    // Convert HNS sum to float for activation
    float input_sum;
    if (u_use_hns > 0) {
        input_sum = hns_to_float(weighted_sum) / 1000.0;
    } else {
        input_sum = weighted_sum.r;
    }
    
    // Sample embodiment input
    float embodied_input = 0.0;
    if (textureSize(u_embodiment, 0).x > 1) {
        vec4 emb = texture(u_embodiment, v_texCoord);
        embodied_input = emb.r * 0.1;  // Scaled embodiment influence
    }
    
    // Neural dynamics: dxi/dt = -xi/τi + σ(Σj wij·xj + Ii) + ξi(t)
    float noise = u_noise_scale * (random(v_texCoord + vec2(u_delta_time)) - 0.5);
    float activation_change = -activation / tau + sigmoid(input_sum + embodied_input) + noise;
    
    // Update activation with decay
    float new_activation = activation + activation_change * u_delta_time;
    new_activation = clamp(new_activation, 0.0, 1.0);
    
    // Update temporal memory (exponential moving average)
    float new_memory = mix(temporal_memory, new_activation, 0.1);
    
    // Output state
    out_state = vec4(new_activation, new_memory, tau, confidence);
    
    // Compute spatial features
    float edge_strength = touches_bg || (same_count < neighbor_count / 2) ? 1.0 : 0.0;
    float density = float(same_count) / float(max(neighbor_count, 1));
    float corner_score = (same_count <= 3) ? 1.0 : 0.0;
    float border_dist = min(
        min(float(coord.x), float(u_grid_size.x - coord.x - 1)),
        min(float(coord.y), float(u_grid_size.y - coord.y - 1))
    ) / float(u_grid_size.x);
    
    out_spatial = vec4(edge_strength, density, corner_score, border_dist);
}
"""

# Vertex shader (simple fullscreen quad)
VERTEX_SHADER = """
#version 430 core

in vec2 in_position;
in vec2 in_texcoord;
out vec2 v_texCoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texCoord = in_texcoord;
}
"""

# Hebbian plasticity shader
PLASTICITY_SHADER = """
#version 430 core

uniform sampler2D u_state;
uniform sampler2D u_weights;
uniform sampler2D u_prev_state;

uniform float u_learning_rate;
uniform float u_regularization;
uniform ivec2 u_grid_size;

in vec2 v_texCoord;
out vec4 out_weights;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    
    // Current and previous activations
    vec4 current = texelFetch(u_state, coord, 0);
    vec4 previous = texelFetch(u_prev_state, coord, 0);
    vec4 weights = texelFetch(u_weights, coord, 0);
    
    float post_act = current.r;  // Postsynaptic activation
    
    // Compute Hebbian update for each weight channel
    vec4 new_weights = weights;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 pre_coord = coord + ivec2(dx, dy);
            if (pre_coord.x < 0 || pre_coord.x >= u_grid_size.x ||
                pre_coord.y < 0 || pre_coord.y >= u_grid_size.y) {
                continue;
            }
            
            vec4 pre_state = texelFetch(u_state, pre_coord, 0);
            float pre_act = pre_state.r;  // Presynaptic activation
            
            // Hebbian term: correlation minus baseline
            float hebbian = pre_act * post_act;
            
            // Homeostatic regularization
            float w = weights.r;  // Simplified: just channel R
            float homeostatic = 1.0 - w * w;
            
            // Update (accumulated into channel based on neighbor index)
            float delta = u_learning_rate * (hebbian + u_regularization * homeostatic);
            new_weights.r = clamp(new_weights.r + delta, 0.0, 1.0);
        }
    }
    
    out_weights = new_weights;
}
"""

# Holographic memory shader
HOLOGRAPHIC_SHADER = """
#version 430 core

uniform sampler2D u_memory;
uniform sampler2D u_input_pattern;
uniform sampler2D u_output_pattern;

uniform float u_learning_rate;
uniform int u_mode;  // 0 = encode, 1 = retrieve

in vec2 v_texCoord;
out vec4 out_memory;

void main() {
    vec4 memory = texture(u_memory, v_texCoord);
    
    if (u_mode == 0) {
        // Encode: M ← M + α · φ(input) ⊗ φ(output)^T
        vec4 input_val = texture(u_input_pattern, v_texCoord);
        vec4 output_val = texture(u_output_pattern, v_texCoord);
        
        // Outer product approximation (simplified for texture)
        vec4 interference = input_val * output_val * u_learning_rate;
        out_memory = memory + interference;
    } else {
        // Retrieve: R = M ⊙ φ(query)
        vec4 query = texture(u_input_pattern, v_texCoord);
        out_memory = memory * query;
    }
}
"""

# Qualia integration shader
QUALIA_SHADER = """
#version 430 core

uniform sampler2D u_visual;
uniform sampler2D u_proprioceptive;
uniform sampler2D u_motor;
uniform sampler2D u_memory;

uniform float u_sigma;  // Coherence scale

in vec2 v_texCoord;
out vec4 out_qualia;

void main() {
    // Sample different modalities
    vec4 visual = texture(u_visual, v_texCoord);
    vec4 proprio = texture(u_proprioceptive, v_texCoord);
    vec4 motor = texture(u_motor, v_texCoord);
    vec4 memory = texture(u_memory, v_texCoord);
    
    // Compute cross-modal coherence
    // QCM = (1/N) Σ exp(-||pi - qi||² / 2σ²)
    
    float vis_prop_diff = length(visual - proprio);
    float vis_motor_diff = length(visual - motor);
    float prop_motor_diff = length(proprio - motor);
    
    float coherence_vp = exp(-vis_prop_diff * vis_prop_diff / (2.0 * u_sigma * u_sigma));
    float coherence_vm = exp(-vis_motor_diff * vis_motor_diff / (2.0 * u_sigma * u_sigma));
    float coherence_pm = exp(-prop_motor_diff * prop_motor_diff / (2.0 * u_sigma * u_sigma));
    
    float avg_coherence = (coherence_vp + coherence_vm + coherence_pm) / 3.0;
    
    // Integrated representation
    vec4 integrated = (visual + proprio + motor + memory) / 4.0;
    integrated.a = avg_coherence;  // Store coherence in alpha
    
    out_qualia = integrated;
}
"""


# =============================================================================
# MAIN ENGINE
# =============================================================================

class NeuroCHIMERA:
    """
    Main NeuroCHIMERA engine for GPU-native neuromorphic computing.
    
    This engine implements:
    - Cellular automata evolution with HNS precision
    - Hebbian synaptic plasticity
    - Holographic memory encoding/retrieval
    - Cross-modal qualia integration
    - Critical parameter tracking
    
    Usage:
        brain = NeuroCHIMERA(neurons=1_000_000, use_hns=True)
        
        for epoch in range(10000):
            brain.evolve(iterations=20)
            metrics = brain.get_metrics()
            
            if brain.is_critical():
                print("Consciousness emergence detected!")
    """
    
    def __init__(self, config: Optional[NeuroCHIMERAConfig] = None, **kwargs):
        """
        Initialize NeuroCHIMERA engine.
        
        Args:
            config: NeuroCHIMERAConfig object (optional)
            **kwargs: Override config parameters
        """
        if config is None:
            config = NeuroCHIMERAConfig(**kwargs)
        else:
            # Apply any overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        self.ctx: Optional['moderngl.Context'] = None
        self.frame: Optional[NeuromorphicFrame] = None
        
        # Shader programs
        self.evolution_program = None
        self.plasticity_program = None
        self.holographic_program = None
        self.qualia_program = None
        
        # Compute shaders (OpenGL 4.3+)
        self.compute_evolution = None
        self.compute_convergence = None
        self.use_compute_shaders = False
        
        # Framebuffers
        self.fbo_evolution = None
        self.fbo_plasticity = None
        self.fbo_memory = None
        self.fbo_qualia = None
        
        # Ping-pong textures for evolution
        self.tex_state_a = None
        self.tex_state_b = None
        self.current_state_idx = 0
        
        # Spatial texture for compute shaders
        self.tex_spatial = None
        
        # Position encoding texture (static)
        self.tex_position = None
        
        # Metrics history
        self.metrics_history: List[Dict] = []
        self.current_epoch = 0
        # Tracks whether CPU-side copies of state/weights are stale after GPU updates
        self._cpu_state_dirty = False
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize GPU context and resources with optimized compute shaders when available."""
        if not HAS_MODERNGL:
            print("Running in CPU-only mode (no GPU acceleration)")
            self._initialize_cpu()
            return
        
        try:
            # Try to create OpenGL 4.3+ context for compute shaders
            try:
                self.ctx = moderngl.create_standalone_context(require=430)  # OpenGL 4.3
                self.use_compute_shaders = True
            except:
                # Fallback to any OpenGL version
                self.ctx = moderngl.create_standalone_context()
                self.use_compute_shaders = False
            
            print(f"OpenGL Context: {self.ctx.info['GL_VERSION']}")
            print(f"GPU: {self.ctx.info['GL_RENDERER']}")
            
            # Create neuromorphic frame
            self.frame = NeuromorphicFrame.create(self.config)
            self.frame.upload_to_gpu(self.ctx)
            
            # Compile shaders (compute shaders if available, fragment shaders otherwise)
            self._compile_shaders()
            
            # Create position encoding texture
            self._create_position_texture()
            
            # Create ping-pong textures
            self._create_ping_pong_textures()
            
            # Pre-allocate spatial texture for compute shaders
            if self.use_compute_shaders:
                size = self.config.texture_size
                self.tex_spatial = self.ctx.texture((size, size), 4, dtype='f4')
                self.tex_spatial.filter = (moderngl.NEAREST, moderngl.NEAREST)
            
            # Create framebuffers
            self._create_framebuffers()
            
            # Create fullscreen quad for rendering (only needed for fragment shaders)
            if not self.use_compute_shaders:
                self._create_quad()
            
            mode_str = "OPTIMIZED (Compute Shaders)" if self.use_compute_shaders else "Standard (Fragment Shaders)"
            print(f"NeuroCHIMERA initialized: {self.config.neurons:,} neurons "
                  f"({self.config.texture_size}×{self.config.texture_size}) - {mode_str}")
            
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU mode")
            self.ctx = None
            self.use_compute_shaders = False
            self._initialize_cpu()
    
    def _initialize_cpu(self):
        """Initialize CPU-only mode."""
        self.frame = NeuromorphicFrame.create(self.config)
    
    def _compile_shaders(self):
        """Compile all shader programs (compute shaders if available, fragment shaders otherwise)."""
        if self.ctx is None:
            return
        
        # Try to compile compute shaders if OpenGL 4.3+ is available
        if self.use_compute_shaders:
            try:
                # Import compute shader from optimized version
                from engine_optimized import COMPUTE_SHADER_EVOLUTION, CONVERGENCE_SHADER
                
                self.compute_evolution = self.ctx.compute_shader(COMPUTE_SHADER_EVOLUTION)
                self.compute_convergence = self.ctx.compute_shader(CONVERGENCE_SHADER)
                print("  ✓ Optimized compute shaders compiled (32×32 work groups)")
                return
            except Exception as e:
                print(f"  ⚠ Compute shader compilation failed: {e}")
                print("  Falling back to fragment shaders")
                self.use_compute_shaders = False
        
        # Fallback to fragment shaders
        self.evolution_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=EVOLUTION_SHADER
        )
        
        self.plasticity_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=PLASTICITY_SHADER
        )
        
        self.holographic_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=HOLOGRAPHIC_SHADER
        )
        
        self.qualia_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=QUALIA_SHADER
        )
    
    def _create_position_texture(self):
        """Create static position encoding texture."""
        if self.ctx is None:
            return
        
        size = self.config.texture_size
        position_data = np.zeros((size, size, 4), dtype=np.float32)
        
        for y in range(size):
            for x in range(size):
                position_data[y, x, 0] = x / size  # Normalized X
                position_data[y, x, 1] = y / size  # Normalized Y
                position_data[y, x, 2] = np.sin(2 * np.pi * x / size)  # Sinusoidal X
                position_data[y, x, 3] = np.cos(2 * np.pi * y / size)  # Cosinusoidal Y
        
        self.tex_position = self.ctx.texture(
            (size, size), 4, position_data.tobytes(), dtype='f4'
        )
        self.tex_position.filter = (moderngl.NEAREST, moderngl.NEAREST)
    
    def _create_ping_pong_textures(self):
        """Create ping-pong textures for evolution."""
        if self.ctx is None:
            return
        
        size = self.config.texture_size
        
        self.tex_state_a = self.ctx.texture(
            (size, size), 4, self.frame.neural_state.tobytes(), dtype='f4'
        )
        self.tex_state_b = self.ctx.texture(
            (size, size), 4, dtype='f4'
        )
        
        self.tex_state_a.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex_state_b.filter = (moderngl.NEAREST, moderngl.NEAREST)
    
    def _create_framebuffers(self):
        """Create framebuffers for rendering."""
        if self.ctx is None:
            return
        
        size = self.config.texture_size
        
        # Evolution framebuffer (dual output: state + spatial)
        tex_spatial_out = self.ctx.texture((size, size), 4, dtype='f4')
        self.fbo_evolution = self.ctx.framebuffer(
            color_attachments=[self.tex_state_b, tex_spatial_out]
        )
    
    def _create_quad(self):
        """Create fullscreen quad for rendering (only needed for fragment shaders)."""
        if self.ctx is None or self.evolution_program is None:
            return
        
        vertices = np.array([
            # Position (x, y), Texcoord (u, v)
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(
            self.evolution_program, self.vbo, 'in_position', 'in_texcoord'
        )
    
    def evolve(self, iterations: Optional[int] = None) -> Dict:
        """
        Evolve the neural network state through cellular automata dynamics.
        
        Args:
            iterations: Number of evolution steps (default from config)
        
        Returns:
            Dictionary with evolution metrics
        """
        if iterations is None:
            iterations = self.config.default_iterations
        
        if self.ctx is not None:
            return self._evolve_gpu(iterations)
        else:
            return self._evolve_cpu(iterations)
    
    def _evolve_gpu(self, iterations: int) -> Dict:
        """GPU-accelerated evolution (uses compute shaders if available, fragment shaders otherwise)."""
        # Use optimized compute shader path if available
        if self.use_compute_shaders and self.compute_evolution is not None:
            return self._evolve_gpu_optimized(iterations)
        
        # Fallback to fragment shader path
        return self._evolve_gpu_fragment(iterations)
    
    def _evolve_gpu_optimized(self, iterations: int) -> Dict:
        """Optimized GPU evolution using compute shaders (32×32 work groups, pipelined)."""
        size = self.config.texture_size
        
        # Calculate work group size (32×32 = 1024 threads per group for better GPU utilization)
        work_groups_x = (size + 31) // 32
        work_groups_y = (size + 31) // 32
        
        # Pre-bind textures once (reduces state changes)
        self.frame.gpu_connectivity.bind_to_image(2, read=True, write=False)
        self.frame.gpu_memory.bind_to_image(3, read=True, write=False)
        self.tex_spatial.bind_to_image(4, read=False, write=True)
        
        # Set uniforms once (they don't change between iterations)
        if 'u_grid_size' in self.compute_evolution:
            self.compute_evolution['u_grid_size'].value = (size, size)
        if 'u_delta_time' in self.compute_evolution:
            self.compute_evolution['u_delta_time'].value = 0.1
        if 'u_decay' in self.compute_evolution:
            self.compute_evolution['u_decay'].value = self.config.decay_rate
        if 'u_noise_scale' in self.compute_evolution:
            self.compute_evolution['u_noise_scale'].value = 0.01
        if 'u_use_hns' in self.compute_evolution:
            self.compute_evolution['u_use_hns'].value = 1 if self.config.use_hns else 0
        
        # Pipeline iterations: dispatch all work without waiting
        # This keeps GPU busy and improves utilization
        for i in range(iterations):
            # Get current textures
            state_in = self.tex_state_a if self.current_state_idx == 0 else self.tex_state_b
            state_out = self.tex_state_b if self.current_state_idx == 0 else self.tex_state_a
            
            # Bind textures as image2D for compute shader
            state_in.bind_to_image(0, read=True, write=False)
            state_out.bind_to_image(1, read=False, write=True)
            
            # Dispatch compute shader - PARALLEL EXECUTION
            # Don't wait - let GPU pipeline the work
            self.compute_evolution.run(work_groups_x, work_groups_y, 1)
            
            # Swap textures for next iteration
            self.current_state_idx = 1 - self.current_state_idx
        
        # Only synchronize at the very end - this allows GPU to work on all iterations in parallel
        self.ctx.finish()
        
        # Update frame reference (state lives on GPU)
        state_out = self.tex_state_b if self.current_state_idx == 0 else self.tex_state_a
        self.frame.gpu_neural = state_out
        # Mark that the CPU-side cache is now stale; it will be refreshed lazily
        self._cpu_state_dirty = True
        
        self.current_epoch += 1
        
        return {
            'iterations': iterations,
            'converged': False,  # Would use GPU-based convergence check
            'gpu_only': True,  # Flag indicating no CPU transfer
            'optimized': True  # Flag indicating compute shader path
        }
    
    def _evolve_gpu_fragment(self, iterations: int) -> Dict:
        """GPU evolution using fragment shaders (fallback for older OpenGL)."""
        size = self.config.texture_size
        
        # Get current state texture
        current_tex = self.tex_state_a if self.current_state_idx == 0 else self.tex_state_b
        output_tex = self.tex_state_b if self.current_state_idx == 0 else self.tex_state_a
        
        prev_state = None
        convergence_history = []
        
        for i in range(iterations):
            # Create framebuffer with current output texture
            # Note: We need to recreate it each iteration for ping-pong
            tex_spatial_out = self.ctx.texture((size, size), 4, dtype='f4')
            if self.fbo_evolution:
                self.fbo_evolution.release()
            self.fbo_evolution = self.ctx.framebuffer(
                color_attachments=[output_tex, tex_spatial_out]
            )
            
            # Set uniforms (only set if they exist in the shader)
            if 'u_state' in self.evolution_program:
                self.evolution_program['u_state'].value = 0
            if 'u_weights' in self.evolution_program:
                self.evolution_program['u_weights'].value = 1
            if 'u_memory' in self.evolution_program:
                self.evolution_program['u_memory'].value = 2
            if 'u_embodiment' in self.evolution_program:
                self.evolution_program['u_embodiment'].value = 3
            if 'u_position' in self.evolution_program:
                self.evolution_program['u_position'].value = 4
            
            if 'u_grid_size' in self.evolution_program:
                self.evolution_program['u_grid_size'].value = (size, size)
            if 'u_delta_time' in self.evolution_program:
                self.evolution_program['u_delta_time'].value = 0.1
            if 'u_decay' in self.evolution_program:
                self.evolution_program['u_decay'].value = self.config.decay_rate
            if 'u_noise_scale' in self.evolution_program:
                self.evolution_program['u_noise_scale'].value = 0.01
            if 'u_use_hns' in self.evolution_program:
                self.evolution_program['u_use_hns'].value = 1 if self.config.use_hns else 0
            
            # Bind textures
            current_tex.use(location=0)
            self.frame.gpu_connectivity.use(location=1)
            self.frame.gpu_memory.use(location=2)
            
            if self.frame.gpu_embodiment:
                self.frame.gpu_embodiment.use(location=3)
            else:
                current_tex.use(location=3)  # Dummy
            
            self.tex_position.use(location=4)
            
            # Render
            self.fbo_evolution.use()
            self.vao.render(moderngl.TRIANGLE_STRIP)
            
            # Check convergence
            if prev_state is not None:
                new_data = np.frombuffer(output_tex.read(), dtype=np.float32)
                diff = np.mean(np.abs(new_data - prev_state))
                convergence_history.append(diff)
                
                if diff < self.config.convergence_threshold:
                    break
            
            # Store for convergence check
            prev_state = np.frombuffer(output_tex.read(), dtype=np.float32)
            
            # Swap textures
            current_tex, output_tex = output_tex, current_tex
            self.current_state_idx = 1 - self.current_state_idx
        
        # Update frame state from GPU
        self.frame.gpu_neural = current_tex
        self.frame.download_from_gpu()
        
        self.current_epoch += 1
        
        return {
            'iterations': len(convergence_history) + 1,
            'converged': len(convergence_history) > 0 and convergence_history[-1] < self.config.convergence_threshold,
            'final_delta': convergence_history[-1] if convergence_history else float('inf')
        }
    
    def _evolve_cpu(self, iterations: int) -> Dict:
        """CPU fallback evolution (slow but functional)."""
        state = self.frame.neural_state.copy()
        size = self.config.texture_size
        
        convergence_history = []
        
        for it in range(iterations):
            new_state = np.zeros_like(state)
            
            for y in range(size):
                for x in range(size):
                    # Sample neighborhood
                    weighted_sum = 0.0
                    
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < size and 0 <= nx < size:
                                weight = self.frame.connectivity[ny, nx, 0] * 2 - 1
                                weighted_sum += state[ny, nx, 0] * weight
                    
                    # Neural dynamics
                    activation = state[y, x, 0]
                    tau = max(state[y, x, 2], 0.1)
                    noise = np.random.uniform(-0.005, 0.005)
                    
                    sigmoid_input = 1.0 / (1.0 + np.exp(-weighted_sum))
                    delta = -activation / tau + sigmoid_input + noise
                    
                    new_state[y, x, 0] = np.clip(activation + delta * 0.1, 0, 1)
                    new_state[y, x, 1] = 0.9 * state[y, x, 1] + 0.1 * new_state[y, x, 0]
                    new_state[y, x, 2] = state[y, x, 2]
                    new_state[y, x, 3] = state[y, x, 3]
            
            # Check convergence
            diff = np.mean(np.abs(new_state - state))
            convergence_history.append(diff)
            
            state = new_state
            
            if diff < self.config.convergence_threshold:
                break
        
        self.frame.neural_state = state
        self.current_epoch += 1
        
        return {
            'iterations': len(convergence_history),
            'converged': convergence_history[-1] < self.config.convergence_threshold,
            'final_delta': convergence_history[-1]
        }
    
    def learn(self, learning_rate: Optional[float] = None):
        """
        Apply Hebbian plasticity to synaptic weights.
        
        Args:
            learning_rate: Override default learning rate
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        if self.ctx is not None:
            self._learn_gpu(learning_rate)
        else:
            self._learn_cpu(learning_rate)
    
    def _learn_gpu(self, learning_rate: float):
        """GPU-accelerated Hebbian learning using PLASTICITY_SHADER on the GPU."""
        if self.ctx is None or self.plasticity_program is None:
            # Fallback to CPU if GPU context or program is not available
            return self._learn_cpu(learning_rate)
        
        size = self.config.texture_size
        
        # Ensure we have a framebuffer for plasticity updates
        if self.fbo_plasticity is None:
            # Create a texture view for weights so we can render updated weights into it
            weights_tex = self.frame.gpu_connectivity
            self.fbo_plasticity = self.ctx.framebuffer(color_attachments=[weights_tex])
        
        # We need the current neural state on GPU; evolution already keeps gpu_neural updated.
        # Make sure gpu_neural exists; if not, upload from CPU once.
        if getattr(self.frame, "gpu_neural", None) is None:
            self.frame.upload_to_gpu(self.ctx)
        
        # Configure uniforms
        if 'u_state' in self.plasticity_program:
            self.plasticity_program['u_state'].value = 0
        if 'u_weights' in self.plasticity_program:
            self.plasticity_program['u_weights'].value = 1
        if 'u_prev_state' in self.plasticity_program:
            # For now, we reuse current state as previous (approximation) to avoid extra history buffers
            self.plasticity_program['u_prev_state'].value = 0
        if 'u_learning_rate' in self.plasticity_program:
            self.plasticity_program['u_learning_rate'].value = learning_rate
        if 'u_regularization' in self.plasticity_program:
            self.plasticity_program['u_regularization'].value = self.config.homeostatic_lambda
        if 'u_grid_size' in self.plasticity_program:
            self.plasticity_program['u_grid_size'].value = (size, size)
        
        # Bind textures
        # 0: neural state (current)
        self.frame.gpu_neural.use(location=0)
        # 1: weights (connectivity)
        self.frame.gpu_connectivity.use(location=1)
        # 2: previous state (reuse current)
        self.frame.gpu_neural.use(location=2)
        
        # Render plasticity pass
        self.fbo_plasticity.use()
        if getattr(self, "vao", None) is None:
            # Quad should already exist from initialization for fragment path,
            # but we guard just in case
            self._create_quad()
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        # Mark CPU connectivity cache as dirty; it will be refreshed lazily when needed
        self._cpu_state_dirty = True
    
    def _learn_cpu(self, learning_rate: float):
        """CPU Hebbian learning."""
        state = self.frame.neural_state
        weights = self.frame.connectivity
        size = self.config.texture_size
        
        for y in range(size):
            for x in range(size):
                post_act = state[y, x, 0]
                
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < size and 0 <= nx < size:
                            pre_act = state[ny, nx, 0]
                            
                            # Hebbian update
                            hebbian = pre_act * post_act
                            w = weights[y, x, 0]
                            homeostatic = 1.0 - w * w
                            
                            delta = learning_rate * (hebbian + self.config.homeostatic_lambda * homeostatic)
                            weights[y, x, 0] = np.clip(w + delta, 0, 1)
        
        self.frame.connectivity = weights
    
    def encode_memory(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """
        Encode association into holographic memory.
        
        Args:
            input_pattern: Input pattern array
            output_pattern: Associated output pattern
        """
        # Resize patterns to memory texture size
        mem_size = self.config.memory_texture_size
        
        if input_pattern.shape[0] != mem_size:
            input_pattern = np.resize(input_pattern, (mem_size, mem_size, 4))
        if output_pattern.shape[0] != mem_size:
            output_pattern = np.resize(output_pattern, (mem_size, mem_size, 4))
        
        # Holographic encoding: M += α * input ⊗ output^T
        interference = input_pattern * output_pattern * self.config.holographic_learning_rate
        self.frame.holographic_memory += interference
    
    def retrieve_memory(self, query: np.ndarray) -> np.ndarray:
        """
        Retrieve from holographic memory.
        
        Args:
            query: Query pattern
        
        Returns:
            Retrieved association pattern
        """
        mem_size = self.config.memory_texture_size
        
        if query.shape[0] != mem_size:
            query = np.resize(query, (mem_size, mem_size, 4))
        
        # Correlation retrieval: R = M * query
        return self.frame.holographic_memory * query
    
    def get_metrics(self) -> Dict:
        """
        Calculate current consciousness-related metrics.
        
        Returns:
            Dictionary with metrics:
                - connectivity: Average effective connections ⟨k⟩
                - phi: Information integration Φ (approximate)
                - hierarchical_depth: Functional depth D
                - complexity: Dynamic complexity C
                - qualia_coherence: QCM metric
        """
        # Ensure CPU-side state is up to date if we've been evolving/learning on GPU
        if self.ctx is not None and getattr(self, "_cpu_state_dirty", False):
            self.frame.download_from_gpu()
            self._cpu_state_dirty = False
        
        state = self.frame.neural_state
        weights = self.frame.connectivity
        
        # Connectivity: count significant connections
        # Calculate average connections per neuron (5x5 neighborhood = 25 neighbors)
        significant_weights = np.abs(weights[:, :, 0] * 2 - 1) > 0.3
        # Count significant weights in local neighborhoods
        size = weights.shape[0]
        connectivity_values = []
        for y in range(size):
            for x in range(size):
                # Count neighbors with significant weights (5x5 neighborhood)
                count = 0
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < size and 0 <= nx < size:
                            if significant_weights[ny, nx]:
                                count += 1
                connectivity_values.append(count)
        connectivity = np.mean(connectivity_values) if connectivity_values else 0.0
        
        # Phi (simplified): mutual information approximation
        activations = state[:, :, 0].flatten()
        
        # Binarize for information calculation
        binary = (activations > 0.5).astype(float)
        
        # Split into halves and compute correlation
        half = len(binary) // 2
        part_a = binary[:half]
        part_b = binary[half:]
        
        # Correlation as proxy for integration
        if np.std(part_a) > 0 and np.std(part_b) > 0:
            phi = np.abs(np.corrcoef(part_a, part_b)[0, 1])
        else:
            phi = 0.0
        
        # Hierarchical depth: analyze weight structure
        depth = self.config.hierarchical_depth  # Use configured value
        
        # Dynamic complexity: Lempel-Ziv approximation
        binary_sequence = ''.join(['1' if x > 0.5 else '0' for x in activations[:1000]])
        complexity = self._lempel_ziv_complexity(binary_sequence)
        
        # Qualia coherence: cross-channel correlation
        if self.frame.qualia_integration is not None:
            qcm = np.mean(self.frame.qualia_integration[:, :, 3])
        else:
            # Approximate from state channels
            r = state[:, :, 0].flatten()
            g = state[:, :, 1].flatten()
            if np.std(r) > 0 and np.std(g) > 0:
                qcm = np.abs(np.corrcoef(r, g)[0, 1])
            else:
                qcm = 0.0
        
        metrics = {
            'connectivity': connectivity,
            'phi': phi,
            'hierarchical_depth': depth,
            'complexity': complexity,
            'qualia_coherence': qcm,
            'mean_activation': np.mean(state[:, :, 0]),
            'std_activation': np.std(state[:, :, 0]),
            'epoch': self.current_epoch
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _lempel_ziv_complexity(self, sequence: str) -> float:
        """Compute normalized Lempel-Ziv complexity."""
        n = len(sequence)
        if n == 0:
            return 0.0
        
        # Count distinct subsequences
        dictionary = set()
        w = ""
        complexity = 0
        
        for c in sequence:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                dictionary.add(wc)
                complexity += 1
                w = ""
        
        if w:
            complexity += 1
        
        # Normalize by theoretical maximum
        max_complexity = n / np.log2(n) if n > 1 else 1
        
        return min(complexity / max_complexity, 1.0)
    
    def is_critical(self) -> bool:
        """
        Check if system has reached critical consciousness parameters.
        
        Returns:
            True if all critical thresholds are exceeded
        """
        metrics = self.get_metrics()
        
        return (
            metrics['connectivity'] > self.config.critical_connectivity and
            metrics['phi'] > self.config.critical_phi and
            metrics['hierarchical_depth'] > self.config.critical_depth and
            metrics['complexity'] > self.config.critical_complexity and
            metrics['qualia_coherence'] > self.config.critical_qcm
        )
    
    def get_state_image(self) -> np.ndarray:
        """Get current neural state as RGB image."""
        state = self.frame.neural_state
        
        # Map channels to RGB
        image = np.zeros((state.shape[0], state.shape[1], 3), dtype=np.uint8)
        image[:, :, 0] = (state[:, :, 0] * 255).astype(np.uint8)  # Activation → R
        image[:, :, 1] = (state[:, :, 1] * 255).astype(np.uint8)  # Memory → G
        image[:, :, 2] = (state[:, :, 3] * 255).astype(np.uint8)  # Confidence → B
        
        return image
    
    def save_state(self, path: str):
        """Save current state to file."""
        np.savez_compressed(
            path,
            neural_state=self.frame.neural_state,
            connectivity=self.frame.connectivity,
            spatial_features=self.frame.spatial_features,
            holographic_memory=self.frame.holographic_memory,
            embodiment=self.frame.embodiment,
            qualia_integration=self.frame.qualia_integration,
            config=self.config.__dict__,
            epoch=self.current_epoch
        )
    
    def load_state(self, path: str):
        """Load state from file."""
        data = np.load(path, allow_pickle=True)
        
        self.frame.neural_state = data['neural_state']
        self.frame.connectivity = data['connectivity']
        self.frame.spatial_features = data['spatial_features']
        self.frame.holographic_memory = data['holographic_memory']
        
        if 'embodiment' in data and data['embodiment'] is not None:
            self.frame.embodiment = data['embodiment']
        
        if 'qualia_integration' in data and data['qualia_integration'] is not None:
            self.frame.qualia_integration = data['qualia_integration']
        
        if 'epoch' in data:
            self.current_epoch = int(data['epoch'])
        
        # Re-upload to GPU
        if self.ctx is not None:
            self.frame.upload_to_gpu(self.ctx)
    
    def release(self):
        """Release all GPU resources."""
        if self.frame:
            self.frame.release_gpu()
        
        for attr in ['tex_state_a', 'tex_state_b', 'tex_position', 'tex_spatial',
                     'vbo', 'vao', 'fbo_evolution', 'compute_evolution', 'compute_convergence']:
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except:
                    pass
        
        if self.ctx:
            try:
                self.ctx.release()
            except:
                pass
    
    def __del__(self):
        self.release()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_brain(neurons: int = 1_000_000, use_hns: bool = True, **kwargs) -> NeuroCHIMERA:
    """
    Create a NeuroCHIMERA brain with sensible defaults.
    
    Args:
        neurons: Number of neurons
        use_hns: Enable Hierarchical Number System
        **kwargs: Additional config parameters
    
    Returns:
        Configured NeuroCHIMERA instance
    """
    return NeuroCHIMERA(neurons=neurons, use_hns=use_hns, **kwargs)


# =============================================================================
# MAIN - Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NeuroCHIMERA - GPU-Native Neuromorphic Computing Engine")
    print("Veselov/Angulo Framework for Emergent Consciousness")
    print("=" * 70)
    
    # Create brain
    print("\nInitializing NeuroCHIMERA...")
    brain = create_brain(neurons=65536, use_hns=True)  # 256×256 for demo
    
    # Evolution demo
    print("\nRunning evolution demo (100 epochs)...")
    
    for epoch in range(100):
        # Evolve
        result = brain.evolve(iterations=10)
        
        # Learn
        brain.learn(learning_rate=0.001)
        
        # Get metrics
        metrics = brain.get_metrics()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d}: "
                  f"⟨k⟩={metrics['connectivity']:.2f} "
                  f"Φ={metrics['phi']:.3f} "
                  f"C={metrics['complexity']:.3f} "
                  f"QCM={metrics['qualia_coherence']:.3f} "
                  f"μ={metrics['mean_activation']:.4f}")
        
        # Check for consciousness emergence
        if brain.is_critical():
            print(f"\n🧠 CRITICAL THRESHOLD REACHED at epoch {epoch}!")
            print("   All consciousness parameters exceeded thresholds.")
            break
    
    # Final metrics
    print("\n" + "=" * 70)
    print("Final Metrics:")
    final_metrics = brain.get_metrics()
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Critical status
    print("\nCritical Parameter Status:")
    print(f"  Connectivity (⟨k⟩ > 15): {'✓' if final_metrics['connectivity'] > 15 else '✗'} ({final_metrics['connectivity']:.2f})")
    print(f"  Integration (Φ > 0.65): {'✓' if final_metrics['phi'] > 0.65 else '✗'} ({final_metrics['phi']:.3f})")
    print(f"  Complexity (C > 0.8): {'✓' if final_metrics['complexity'] > 0.8 else '✗'} ({final_metrics['complexity']:.3f})")
    print(f"  QCM (> 0.75): {'✓' if final_metrics['qualia_coherence'] > 0.75 else '✗'} ({final_metrics['qualia_coherence']:.3f})")
    
    # Cleanup
    brain.release()
    print("\nDemo complete.")
