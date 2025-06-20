#ifndef ATTENTION_RECORDER_H
#define ATTENTION_RECORDER_H

#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <memory>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "ggml.h"

// Forward declaration for GGML tensor (adjust include path as needed)
struct ggml_tensor;

// Structure to hold attention weights for a single layer
struct LayerAttentionWeights {
    int layer_id;
    ggml_tensor* attention_tensor;  // Reference to the kq tensor
    std::string layer_name;
    
    // Cached properties (filled after execution)
    int64_t n_kv = 0;
    int64_t n_tokens = 0; 
    int64_t n_head = 0;
    int64_t n_batch = 0;
    
    /**
     * @brief Constructor for LayerAttentionWeights
     * @param id Layer identifier
     * @param tensor Pointer to the attention tensor (kq)
     * @param name Optional layer name for debugging
     */
    LayerAttentionWeights(int id, ggml_tensor* tensor, const std::string& name = "");
    
    /**
     * @brief Get attention weights data (call after graph execution)
     * @return Pointer to float data or nullptr if not available
     */
    const float* get_data() const;
    
    /**
     * @brief Update cached dimensions from tensor
     */
    void update_dimensions();
    
    /**
     * @brief Get total number of elements in the attention tensor
     * @return Total number of float elements
     */
    size_t get_total_elements() const;
    
    /**
     * @brief Check if tensor data is available
     * @return true if data is accessible
     */
    bool is_data_available() const;
};

/**
 * @brief Main class for recording and managing attention weights across all layers
 * 
 * This class provides functionality to:
 * - Record attention weight tensors during graph construction
 * - Process and analyze attention weights after graph execution
 * - Save attention weights to files
 * - Extract specific attention patterns
 */
class AttentionWeightsRecorder {
private:
    std::vector<LayerAttentionWeights> layer_weights;
    std::unordered_map<int, size_t> layer_id_to_index;
    bool recording_enabled = true;
    
    // Optional callback for real-time processing
    std::function<void(const LayerAttentionWeights&)> process_callback;
    
public:
    /**
     * @brief Default constructor
     */
    AttentionWeightsRecorder() = default;
    
    /**
     * @brief Destructor
     */
    ~AttentionWeightsRecorder() = default;
    
    // Disable copy constructor and assignment operator
    AttentionWeightsRecorder(const AttentionWeightsRecorder&) = delete;
    AttentionWeightsRecorder& operator=(const AttentionWeightsRecorder&) = delete;
    
    /**
     * @brief Enable or disable recording
     * @param enabled Whether to enable recording
     */
    void set_recording(bool enabled);
    
    /**
     * @brief Check if recording is currently enabled
     * @return true if recording is enabled
     */
    bool is_recording() const;
    
    /**
     * @brief Set callback function for processing each layer's attention weights
     * @param callback Function to call for each layer after execution
     */
    void set_process_callback(std::function<void(const LayerAttentionWeights&)> callback);
    
    /**
     * @brief Record attention weights for a layer (call during graph building)
     * @param layer_id Unique identifier for the layer
     * @param kq_tensor Pointer to the attention weights tensor
     * @param layer_name Optional descriptive name for the layer
     */
    void record_layer(int layer_id, ggml_tensor* kq_tensor, const std::string& layer_name = "");
    
    /**
     * @brief Process all recorded attention weights (call after graph execution)
     * Updates dimensions and calls process callback for each layer
     */
    void process_all_layers();
    
    /**
     * @brief Get attention weights for a specific layer
     * @param layer_id Layer identifier
     * @return Pointer to LayerAttentionWeights or nullptr if not found
     */
    const LayerAttentionWeights* get_layer(int layer_id) const;
    
    /**
     * @brief Get all recorded layers
     * @return Const reference to vector of all layers
     */
    const std::vector<LayerAttentionWeights>& get_all_layers() const;
    
    /**
     * @brief Clear all recorded data
     */
    void clear();
    
    /**
     * @brief Get number of recorded layers
     * @return Number of layers currently recorded
     */
    size_t size() const;
    
    /**
     * @brief Check if recorder has any recorded layers
     * @return true if no layers recorded
     */
    bool empty() const;
    
    /**
     * @brief Print summary of all recorded layers to stdout
     */
    void print_summary() const;
    
    /**
     * @brief Save all attention weights to binary files
     * @param base_path Base path for output files (without extension)
     * File format: [base_path]_layer_[id].bin
     * Each file contains: int64_t dimensions (4x) + float data
     */
    void save_to_files(const std::string& base_path = "attention_weights") const;
    
    /**
     * @brief Load attention weights from binary file
     * @param filepath Path to binary file
     * @param layer_id Layer ID to assign to loaded data
     * @return true if successfully loaded
     */
    bool load_from_file(const std::string& filepath, int layer_id);
    
    /**
     * @brief Extract attention pattern for specific head and token
     * @param layer_id Layer identifier
     * @param head_idx Head index (0-based)
     * @param token_idx Token index (0-based)
     * @return Vector of attention weights for all keys, empty if invalid parameters
     */
    std::vector<float> get_attention_pattern(int layer_id, int head_idx, int token_idx) const;
    
    /**
     * @brief Get attention weights for a specific position
     * @param layer_id Layer identifier
     * @param head_idx Head index
     * @param token_idx Token index (query)
     * @param key_idx Key index
     * @return Attention weight value, or 0.0f if invalid parameters
     */
    float get_attention_weight(int layer_id, int head_idx, int token_idx, int key_idx) const;
    
    /**
     * @brief Find maximum attention weight across all positions in a layer
     * @param layer_id Layer identifier
     * @param out_head Optional pointer to store head index of maximum
     * @param out_token Optional pointer to store token index of maximum
     * @param out_key Optional pointer to store key index of maximum
     * @return Maximum attention weight value, or 0.0f if layer not found
     */
    float find_max_attention(int layer_id, int* out_head = nullptr, 
                           int* out_token = nullptr, int* out_key = nullptr) const;
    
    // /**
    //  * @brief Export attention weights to CSV format
    //  * @param layer_id Layer identifier
    //  * @param filepath Output CSV file path
    //  * @param include_headers Whether to include column headers
    //  * @return true if successfully exported
    //  */
    // bool export_to_csv(int layer_id, const std::string& filepath, bool include_headers = true) const;
    
    // /**
    //  * @brief Get memory usage statistics
    //  * @return Estimated memory usage in bytes
    //  */
    // size_t get_memory_usage() const;
    
    // /**
    //  * @brief Validate all recorded tensors have valid data
    //  * @return Number of layers with valid data
    //  */
    // size_t validate_data() const;
};

// Global instance declaration (define in your .cpp file)
extern AttentionWeightsRecorder g_attention_recorder;

// Utility functions

/**
 * @brief Initialize global attention recorder with default settings
 */
void init_attention_recorder();

/**
 * @brief Cleanup and reset global attention recorder
 */
void cleanup_attention_recorder();

/**
 * @brief Convenience function to start recording attention weights
 * @param enable_callback Whether to enable default processing callback
 */
void start_attention_recording(bool enable_callback = true);

/**
 * @brief Convenience function to stop recording and process results
 * @param save_to_files Whether to automatically save results to files
 * @param base_path Base path for saved files (if save_to_files is true)
 */
void stop_attention_recording(bool save_to_files = false, const std::string& base_path = "attention_weights");

/**
 * @brief Default callback function for processing attention weights
 * Prints summary information for each layer
 * @param layer The layer to process
 */
void default_attention_callback(const LayerAttentionWeights& layer);

/**
 * @brief Analyze attention patterns across all recorded layers
 * Prints comprehensive analysis including max weights, entropy, etc.
 */
void analyze_all_attention_patterns();

#endif // ATTENTION_RECORDER_H