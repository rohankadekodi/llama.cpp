#include "attention_recorder.h"
#include <iostream>
#include <fstream>
#include <cmath>

// Include your GGML headers here
// #include "ggml.h"

// Global instance definition
AttentionWeightsRecorder g_attention_recorder;

// LayerAttentionWeights implementation
LayerAttentionWeights::LayerAttentionWeights(int id, ggml_tensor* tensor, const std::string& name)
    : layer_id(id), attention_tensor(tensor), layer_name(name) {
}

const float* LayerAttentionWeights::get_data() const {
    return attention_tensor && attention_tensor->data ? 
           (const float*)attention_tensor->data : nullptr;
}

void LayerAttentionWeights::update_dimensions() {
    if (attention_tensor) {
        n_kv = attention_tensor->ne[0];
        n_tokens = attention_tensor->ne[1];
        n_head = attention_tensor->ne[2];
        n_batch = attention_tensor->ne[3];
    }
}

size_t LayerAttentionWeights::get_total_elements() const {
    return n_kv * n_tokens * n_head * n_batch;
}

bool LayerAttentionWeights::is_data_available() const {
    return attention_tensor && attention_tensor->data;
}

// AttentionWeightsRecorder implementation
void AttentionWeightsRecorder::set_recording(bool enabled) {
    recording_enabled = enabled;
}

bool AttentionWeightsRecorder::is_recording() const {
    return recording_enabled;
}

void AttentionWeightsRecorder::set_process_callback(std::function<void(const LayerAttentionWeights&)> callback) {
    process_callback = callback;
}

void AttentionWeightsRecorder::record_layer(int layer_id, ggml_tensor* kq_tensor, const std::string& layer_name) {
    if (!recording_enabled || !kq_tensor) return;
    
    auto it = layer_id_to_index.find(layer_id);
    if (it != layer_id_to_index.end()) {
        // Update existing layer
        layer_weights[it->second].attention_tensor = kq_tensor;
        layer_weights[it->second].layer_name = layer_name;
    } else {
        // Add new layer
        layer_weights.emplace_back(layer_id, kq_tensor, layer_name);
        layer_id_to_index[layer_id] = layer_weights.size() - 1;
    }
}

void AttentionWeightsRecorder::process_all_layers() {
    if (!recording_enabled) return;
    
    for (auto& layer : layer_weights) {
        layer.update_dimensions();
        
        if (process_callback) {
            process_callback(layer);
        }
    }
}

const LayerAttentionWeights* AttentionWeightsRecorder::get_layer(int layer_id) const {
    auto it = layer_id_to_index.find(layer_id);
    return it != layer_id_to_index.end() ? &layer_weights[it->second] : nullptr;
}

const std::vector<LayerAttentionWeights>& AttentionWeightsRecorder::get_all_layers() const {
    return layer_weights;
}

void AttentionWeightsRecorder::clear() {
    layer_weights.clear();
    layer_id_to_index.clear();
}

size_t AttentionWeightsRecorder::size() const {
    return layer_weights.size();
}

bool AttentionWeightsRecorder::empty() const {
    return layer_weights.empty();
}

void AttentionWeightsRecorder::print_summary() const {
    printf("=== Attention Weights Summary ===\n");
    printf("Total layers recorded: %zu\n", layer_weights.size());
    
    for (const auto& layer : layer_weights) {
        printf("Layer %d", layer.layer_id);
        if (!layer.layer_name.empty()) {
            printf(" (%s)", layer.layer_name.c_str());
        }
        printf(": [%lld, %lld, %lld, %lld] = %zu elements",
               layer.n_kv, layer.n_tokens, layer.n_head, layer.n_batch,
               layer.get_total_elements());
        
        if (!layer.is_data_available()) {
            printf(" [NO DATA]");
        }
        printf("\n");
    }
    printf("================================\n");
}

void AttentionWeightsRecorder::save_to_files(const std::string& base_path) const {
    for (const auto& layer : layer_weights) {
        const float* data = layer.get_data();
        if (!data) {
            printf("Warning: No data available for layer %d, skipping save\n", layer.layer_id);
            continue;
        }
        
        std::string filename = base_path + "_layer_" + std::to_string(layer.layer_id) + ".bin";
        FILE* file = fopen(filename.c_str(), "w");
        if (file) {
            // Write dimensions first in a csv format
            fprintf(file, "%d,%d,%d,%d\n", layer.n_kv, layer.n_tokens, layer.n_head, layer.n_batch);
            // fwrite(&layer.n_kv, sizeof(int64_t), 1, file);
            // fwrite(&layer.n_tokens, sizeof(int64_t), 1, file);
            // fwrite(&layer.n_head, sizeof(int64_t), 1, file);
            // fwrite(&layer.n_batch, sizeof(int64_t), 1, file);
            
            // Write attention data in a csv format
           size_t total_elements = layer.get_total_elements();
           fprintf(file, "total_elements: %lu, dimensions of data = %lu\n", total_elements, sizeof(data));
            for (int i = 0; i < layer.n_kv; i++) {
                for (int j = 0; j < layer.n_tokens; j++) {
                    for (int k = 0; k < layer.n_head; k++) {
                        for (int l = 0; l < layer.n_batch; l++) {
                            fprintf(file, "%f,", data[i * layer.n_tokens * layer.n_head * layer.n_batch + j * layer.n_head * layer.n_batch + k * layer.n_batch + l]);
                        }
                    }
                }
            }
            // size_t total_elements = layer.get_total_elements();            
            // fwrite(data, sizeof(float), total_elements, file);
            fclose(file);
            
            printf("Saved layer %d attention weights to %s\n", layer.layer_id, filename.c_str());
        } else {
            printf("Error: Could not open file %s for writing\n", filename.c_str());
        }
    }
}

std::vector<float> AttentionWeightsRecorder::get_attention_pattern(int layer_id, int head_idx, int token_idx) const {
    const auto* layer = get_layer(layer_id);
    if (!layer || !layer->get_data() || head_idx >= layer->n_head || token_idx >= layer->n_tokens) {
        return {};
    }
    
    const float* data = layer->get_data();
    std::vector<float> pattern;
    pattern.reserve(layer->n_kv);
    
    // Extract attention weights for specific head and token
    size_t offset = head_idx * layer->n_tokens * layer->n_kv + token_idx * layer->n_kv;
    for (int64_t k = 0; k < layer->n_kv; k++) {
        pattern.push_back(data[offset + k]);
    }
    
    return pattern;
}

float AttentionWeightsRecorder::get_attention_weight(int layer_id, int head_idx, int token_idx, int key_idx) const {
    const auto* layer = get_layer(layer_id);
    if (!layer || !layer->get_data() || 
        head_idx >= layer->n_head || token_idx >= layer->n_tokens || key_idx >= layer->n_kv) {
        return 0.0f;
    }
    
    const float* data = layer->get_data();
    size_t idx = head_idx * layer->n_tokens * layer->n_kv + token_idx * layer->n_kv + key_idx;
    return data[idx];
}

float AttentionWeightsRecorder::find_max_attention(int layer_id, int* out_head, int* out_token, int* out_key) const {
    // TODO: Implement this
    return 0.0f;
}


// Utility functions
void init_attention_recorder() {
    g_attention_recorder.clear();
    g_attention_recorder.set_recording(true);
}

void cleanup_attention_recorder() {
    g_attention_recorder.clear();
    g_attention_recorder.set_recording(false);
}

void start_attention_recording(bool enable_callback) {
    g_attention_recorder.set_recording(true);
    g_attention_recorder.clear();
    
    if (enable_callback) {
        g_attention_recorder.set_process_callback(default_attention_callback);
    }
}

void stop_attention_recording(bool save_to_files, const std::string& base_path) {
    g_attention_recorder.process_all_layers();
    g_attention_recorder.print_summary();
    
    if (save_to_files) {
        g_attention_recorder.save_to_files(base_path);
    }
    
    g_attention_recorder.set_recording(false);
}

void default_attention_callback(const LayerAttentionWeights& layer) {
    printf("Processing layer %d (%s): [%lld, %lld, %lld, %lld]\n", 
           layer.layer_id, 
           layer.layer_name.empty() ? "unnamed" : layer.layer_name.c_str(),
           layer.n_kv, layer.n_tokens, layer.n_head, layer.n_batch);
    
    if (layer.get_data()) {
        // Find max attention for this layer
        int max_head, max_token, max_key;
        float max_val = g_attention_recorder.find_max_attention(layer.layer_id, &max_head, &max_token, &max_key);
        printf("  Max attention: %.6f at head=%d, token=%d, key=%d\n", max_val, max_head, max_token, max_key);
    }
}
