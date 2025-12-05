#ifndef __CONTROL_DIT_HPP__
#define __CONTROL_DIT_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"
#include "model.h"
#include "flux.hpp"
#include "stable-diffusion.h"

#define CONTROL_DIT_GRAPH_SIZE 10240

/*
    =================================== InstantX ControlNet Union for Qwen Image ===================================
    Reference: https://huggingface.co/InstantX/Qwen-Image-InstantX-ControlNet-Union
    
    NOTE: This file is currently a stub. ControlNet DiT is disabled in QwenDiffuse.cpp.
    The implementation needs work to properly integrate with the Qwen pipeline.
*/

namespace QwenControlNet {

class ControlNetAttention : public GGMLBlock {
protected:
    int64_t dim_head;
    int64_t num_heads;
    bool flash_attn;

public:
    ControlNetAttention(int64_t query_dim,
                        int64_t dim_head,
                        int64_t num_heads,
                        bool flash_attn = false)
        : dim_head(dim_head), num_heads(num_heads), flash_attn(flash_attn) {
        int64_t inner_dim = dim_head * num_heads;

        blocks["to_q"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, true));
        blocks["to_k"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, true));
        blocks["to_v"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, true));
        blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, 1e-6f));
        blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, 1e-6f));
        blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, query_dim, true));

        blocks["add_q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, true));
        blocks["add_k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, true));
        blocks["add_v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, true));
        blocks["norm_added_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, 1e-6f));
        blocks["norm_added_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, 1e-6f));
        blocks["to_add_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, query_dim, true));
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                                 struct ggml_tensor* img,
                                                                 struct ggml_tensor* txt,
                                                                 struct ggml_tensor* pe) {
        auto to_q = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto to_k = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
        auto to_v = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
        auto norm_q = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
        auto norm_k = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
        auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

        auto add_q_proj = std::dynamic_pointer_cast<Linear>(blocks["add_q_proj"]);
        auto add_k_proj = std::dynamic_pointer_cast<Linear>(blocks["add_k_proj"]);
        auto add_v_proj = std::dynamic_pointer_cast<Linear>(blocks["add_v_proj"]);
        auto norm_added_q = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_added_q"]);
        auto norm_added_k = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_added_k"]);
        auto to_add_out = std::dynamic_pointer_cast<Linear>(blocks["to_add_out"]);

        int64_t n_head = num_heads;
        auto gctx = ctx->ggml_ctx;

        auto q = to_q->forward(ctx, img);
        auto k = to_k->forward(ctx, img);
        auto v = to_v->forward(ctx, img);

        auto add_q = add_q_proj->forward(ctx, txt);
        auto add_k = add_k_proj->forward(ctx, txt);
        auto add_v = add_v_proj->forward(ctx, txt);

        q = ggml_reshape_4d(gctx, q, dim_head, n_head, q->ne[1], q->ne[2]);
        q = norm_q->forward(ctx, q);
        k = ggml_reshape_4d(gctx, k, dim_head, n_head, k->ne[1], k->ne[2]);
        k = norm_k->forward(ctx, k);

        add_q = ggml_reshape_4d(gctx, add_q, dim_head, n_head, add_q->ne[1], add_q->ne[2]);
        add_q = norm_added_q->forward(ctx, add_q);
        add_k = ggml_reshape_4d(gctx, add_k, dim_head, n_head, add_k->ne[1], add_k->ne[2]);
        add_k = norm_added_k->forward(ctx, add_k);

        q = ggml_reshape_3d(gctx, q, dim_head * n_head, q->ne[2], q->ne[3]);
        k = ggml_reshape_3d(gctx, k, dim_head * n_head, k->ne[2], k->ne[3]);
        add_q = ggml_reshape_3d(gctx, add_q, dim_head * n_head, add_q->ne[2], add_q->ne[3]);
        add_k = ggml_reshape_3d(gctx, add_k, dim_head * n_head, add_k->ne[2], add_k->ne[3]);

        q = ggml_concat(gctx, q, add_q, 1);
        k = ggml_concat(gctx, k, add_k, 1);
        v = ggml_concat(gctx, v, add_v, 1);

        auto attn = ggml_ext_attention(gctx, q, k, v, false);

        int64_t n_img = img->ne[1];
        int64_t n_txt = txt->ne[1];

        auto img_attn_out = ggml_view_3d(gctx, attn, attn->ne[0], n_img, attn->ne[2],
                                          attn->nb[1], attn->nb[2], 0);
        auto txt_attn_out = ggml_view_3d(gctx, attn, attn->ne[0], n_txt, attn->ne[2],
                                          attn->nb[1], attn->nb[2], attn->nb[1] * n_img);

        img_attn_out = ggml_cont(gctx, img_attn_out);
        txt_attn_out = ggml_cont(gctx, txt_attn_out);

        img_attn_out = to_out_0->forward(ctx, img_attn_out);
        txt_attn_out = to_add_out->forward(ctx, txt_attn_out);

        return std::make_pair(img_attn_out, txt_attn_out);
    }
};

class ControlNetTransformerBlock : public GGMLBlock {
protected:
    int64_t hidden_size;

public:
    ControlNetTransformerBlock(int64_t hidden_size,
                               int64_t num_attention_heads,
                               int64_t attention_head_dim,
                               bool flash_attn = false)
        : hidden_size(hidden_size) {
        blocks["img_mod.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 6 * hidden_size, true));
        blocks["txt_mod.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 6 * hidden_size, true));
        blocks["img_mlp.net.0"] = std::shared_ptr<GGMLBlock>(new GELU(hidden_size, hidden_size * 4));
        blocks["img_mlp.net.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size * 4, hidden_size, true));
        blocks["txt_mlp.net.0"] = std::shared_ptr<GGMLBlock>(new GELU(hidden_size, hidden_size * 4));
        blocks["txt_mlp.net.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size * 4, hidden_size, true));
        blocks["attn"] = std::shared_ptr<GGMLBlock>(new ControlNetAttention(
            hidden_size, attention_head_dim, num_attention_heads, flash_attn));
    }

    std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                   struct ggml_tensor* img,
                                                   struct ggml_tensor* txt,
                                                   struct ggml_tensor* t_emb,
                                                   struct ggml_tensor* pe) {
        auto img_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["img_mod.1"]);
        auto txt_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["txt_mod.1"]);
        auto img_mlp_0 = std::dynamic_pointer_cast<GELU>(blocks["img_mlp.net.0"]);
        auto img_mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.net.2"]);
        auto txt_mlp_0 = std::dynamic_pointer_cast<GELU>(blocks["txt_mlp.net.0"]);
        auto txt_mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.net.2"]);
        auto attn = std::dynamic_pointer_cast<ControlNetAttention>(blocks["attn"]);

        auto gctx = ctx->ggml_ctx;

        auto img_mod = img_mod_1->forward(ctx, ggml_silu(gctx, t_emb));
        auto txt_mod = txt_mod_1->forward(ctx, ggml_silu(gctx, t_emb));
        
        auto img_mods = ggml_ext_chunk(gctx, img_mod, 6, 0);
        auto txt_mods = ggml_ext_chunk(gctx, txt_mod, 6, 0);

        // Use Flux::modulate which avoids ggml_new_f32 in no_alloc context
        auto img_mod_input = Flux::modulate(gctx, img, img_mods[0], img_mods[1]);
        auto txt_mod_input = Flux::modulate(gctx, txt, txt_mods[0], txt_mods[1]);

        auto [img_attn, txt_attn] = attn->forward(ctx, img_mod_input, txt_mod_input, pe);

        auto img_gate1 = ggml_reshape_3d(gctx, img_mods[2], img_mods[2]->ne[0], 1, img_mods[2]->ne[1]);
        auto txt_gate1 = ggml_reshape_3d(gctx, txt_mods[2], txt_mods[2]->ne[0], 1, txt_mods[2]->ne[1]);
        img = ggml_add(gctx, img, ggml_mul(gctx, img_attn, img_gate1));
        txt = ggml_add(gctx, txt, ggml_mul(gctx, txt_attn, txt_gate1));

        auto img_mod_mlp = Flux::modulate(gctx, img, img_mods[3], img_mods[4]);
        auto txt_mod_mlp = Flux::modulate(gctx, txt, txt_mods[3], txt_mods[4]);

        auto img_mlp_out = img_mlp_2->forward(ctx, img_mlp_0->forward(ctx, img_mod_mlp));
        auto txt_mlp_out = txt_mlp_2->forward(ctx, txt_mlp_0->forward(ctx, txt_mod_mlp));

        auto img_gate2 = ggml_reshape_3d(gctx, img_mods[5], img_mods[5]->ne[0], 1, img_mods[5]->ne[1]);
        auto txt_gate2 = ggml_reshape_3d(gctx, txt_mods[5], txt_mods[5]->ne[0], 1, txt_mods[5]->ne[1]);
        img = ggml_add(gctx, img, ggml_mul(gctx, img_mlp_out, img_gate2));
        txt = ggml_add(gctx, txt, ggml_mul(gctx, txt_mlp_out, txt_gate2));

        return std::make_pair(img, txt);
    }
};

class ControlNetDiTBlock : public GGMLBlock {
protected:
    int64_t hidden_size = 3072;
    int64_t num_layers = 5;
    int64_t num_attention_heads = 24;
    int64_t attention_head_dim = 128;
    int64_t in_channels = 64;
    int64_t joint_attention_dim = 3584;
    int64_t num_control_types = SD_CONTROL_TYPE_COUNT - 1;  // Exclude AUTO (-1)
    bool flash_attn = false;

public:
    ControlNetDiTBlock(int64_t hidden_size = 3072,
                       int64_t num_layers = 5,
                       int64_t num_attention_heads = 24,
                       int64_t attention_head_dim = 128,
                       int64_t in_channels = 64,
                       int64_t joint_attention_dim = 3584,
                       int64_t num_control_types = 8,
                       bool flash_attn = false)
        : hidden_size(hidden_size),
          num_layers(num_layers),
          num_attention_heads(num_attention_heads),
          attention_head_dim(attention_head_dim),
          in_channels(in_channels),
          joint_attention_dim(joint_attention_dim),
          num_control_types(num_control_types),
          flash_attn(flash_attn) {
        
        blocks["time_text_embed.timestep_embedder.linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(256, hidden_size, true));
        blocks["time_text_embed.timestep_embedder.linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true));
        blocks["controlnet_x_embedder"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, hidden_size, true));
        blocks["img_in"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, hidden_size, true));
        blocks["txt_in"] = std::shared_ptr<GGMLBlock>(new Linear(joint_attention_dim, hidden_size, true));
        blocks["txt_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(joint_attention_dim, 1e-6f));
        
        for (int i = 0; i < num_layers; i++) {
            blocks["transformer_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                new ControlNetTransformerBlock(hidden_size, num_attention_heads, attention_head_dim, flash_attn));
            blocks["controlnet_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                new Linear(hidden_size, hidden_size, true));
        }
    }
    
    struct ggml_tensor* get_timestep_embedding(GGMLRunnerContext* ctx, struct ggml_tensor* timesteps) {
        auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["time_text_embed.timestep_embedder.linear_1"]);
        auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["time_text_embed.timestep_embedder.linear_2"]);
        
        auto gctx = ctx->ggml_ctx;
        auto t_emb = ggml_timestep_embedding(gctx, timesteps, 256, 10000);
        t_emb = linear_1->forward(ctx, t_emb);
        t_emb = ggml_silu_inplace(gctx, t_emb);
        t_emb = linear_2->forward(ctx, t_emb);
        
        return t_emb;
    }
    
    std::vector<struct ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                             struct ggml_tensor* control_img,
                                             struct ggml_tensor* img,
                                             struct ggml_tensor* txt,
                                             struct ggml_tensor* timesteps,
                                             struct ggml_tensor* pe,
                                             struct ggml_tensor* task_embedding = nullptr,
                                             int control_type = -1) {
        auto controlnet_x_embedder = std::dynamic_pointer_cast<Linear>(blocks["controlnet_x_embedder"]);
        auto img_in = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
        auto txt_in = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
        auto txt_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"]);
        
        auto gctx = ctx->ggml_ctx;
        
        if (ggml_n_dims(txt) == 4) {
            txt = ggml_reshape_3d(gctx, txt, txt->ne[0], txt->ne[1], txt->ne[2]);
        }
        
        auto t_emb = get_timestep_embedding(ctx, timesteps);
        auto control = controlnet_x_embedder->forward(ctx, control_img);
        auto img_hidden = img_in->forward(ctx, img);
        img_hidden = ggml_add(gctx, img_hidden, control);
        
        // Apply task embedding for Union ControlNet if control_type is specified
        if (task_embedding != nullptr && control_type >= 0 && control_type < num_control_types) {
            // task_embedding shape: [num_control_types, hidden_size]
            // Select the embedding for the specified control type
            auto task_emb = ggml_view_2d(gctx, task_embedding, 
                                          task_embedding->ne[0], 1,
                                          task_embedding->nb[1],
                                          control_type * task_embedding->nb[1]);
            // Broadcast and add to img_hidden
            // img_hidden shape: [hidden_size, seq_len, batch]
            task_emb = ggml_reshape_3d(gctx, task_emb, task_emb->ne[0], 1, 1);
            img_hidden = ggml_add(gctx, img_hidden, task_emb);
        }
        
        auto txt_hidden = txt_norm->forward(ctx, txt);
        txt_hidden = txt_in->forward(ctx, txt_hidden);
        
        std::vector<struct ggml_tensor*> block_outputs;
        
        for (int i = 0; i < num_layers; i++) {
            auto block = std::dynamic_pointer_cast<ControlNetTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
            auto controlnet_block = std::dynamic_pointer_cast<Linear>(blocks["controlnet_blocks." + std::to_string(i)]);
            
            auto result = block->forward(ctx, img_hidden, txt_hidden, t_emb, pe);
            img_hidden = result.first;
            txt_hidden = result.second;
            
            auto control_out = controlnet_block->forward(ctx, img_hidden);
            block_outputs.push_back(control_out);
        }
        
        return block_outputs;
    }
};

struct ControlNetDiT : public GGMLRunner {
    ControlNetDiTBlock control_net;
    int patch_size = 2;
    int num_control_types = 8;  // Union ControlNet types
    
    ggml_backend_buffer_t control_buffer = nullptr;
    ggml_context* control_ctx = nullptr;
    std::vector<struct ggml_tensor*> controls;
    
    // Task embedding for Union ControlNet - loaded from model
    struct ggml_tensor* task_embedding = nullptr;
    int current_control_type = -1;  // Current control type being used (-1 = auto)
    
    ControlNetDiT(ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_types = {},
                  bool flash_attn = false)
        : GGMLRunner(backend, offload_params_to_cpu),
          control_net(3072, 5, 24, 128, 64, 3584, 8, flash_attn) {
        control_net.init(params_ctx, tensor_types, "");
        
        // Create task_embedding tensor in params_ctx
        // Shape: [num_control_types, hidden_size] = [8, 3072]
        task_embedding = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F32, 3072, num_control_types);
    }
    
    void set_control_type(int control_type) {
        if (control_type >= -1 && control_type < num_control_types) {
            current_control_type = control_type;
        }
    }
    
    void set_control_type(sd_control_type_t control_type) {
        set_control_type(static_cast<int>(control_type));
    }
    
    ~ControlNetDiT() override {
        free_control_ctx();
    }
    
    void alloc_control_ctx(std::vector<struct ggml_tensor*> outs) {
        struct ggml_init_params params;
        params.mem_size = static_cast<size_t>(outs.size() * ggml_tensor_overhead()) + 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        control_ctx = ggml_init(params);
        
        controls.resize(outs.size());
        size_t control_buffer_size = 0;
        
        for (size_t i = 0; i < outs.size(); i++) {
            controls[i] = ggml_dup_tensor(control_ctx, outs[i]);
            control_buffer_size += ggml_nbytes(controls[i]);
        }
        
        control_buffer = ggml_backend_alloc_ctx_tensors(control_ctx, runtime_backend);
        LOG_DEBUG("control dit buffer size %.2fMB", control_buffer_size * 1.f / 1024.f / 1024.f);
    }
    
    void free_control_ctx() {
        if (control_buffer != nullptr) {
            ggml_backend_buffer_free(control_buffer);
            control_buffer = nullptr;
        }
        if (control_ctx != nullptr) {
            ggml_free(control_ctx);
            control_ctx = nullptr;
        }
        controls.clear();
    }
    
    std::string get_desc() override {
        return "control_net_dit";
    }
    
    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        control_net.get_param_tensors(tensors, prefix);
        // Add task_embedding for Union ControlNet
        tensors[prefix + "task_embedding"] = task_embedding;
    }
    
    struct ggml_tensor* pad_to_patch_size(struct ggml_context* ctx, struct ggml_tensor* x) {
        int64_t W = x->ne[0];
        int64_t H = x->ne[1];
        int pad_h = (patch_size - H % patch_size) % patch_size;
        int pad_w = (patch_size - W % patch_size) % patch_size;
        x = ggml_pad(ctx, x, pad_w, pad_h, 0, 0);
        return x;
    }
    
    struct ggml_tensor* patchify(struct ggml_context* ctx, struct ggml_tensor* x) {
        int64_t N = x->ne[3];
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];
        int64_t p = patch_size;
        int64_t h = H / patch_size;
        int64_t w = W / patch_size;

        x = ggml_reshape_4d(ctx, x, p, w, p, h * C * N);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx, x, p * p, w * h, C, N);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_3d(ctx, x, p * p * C, w * h, N);
        return x;
    }
    
    struct ggml_tensor* process_latent(struct ggml_context* ctx, struct ggml_tensor* x) {
        x = pad_to_patch_size(ctx, x);
        x = patchify(ctx, x);
        return x;
    }
    
    struct ggml_cgraph* build_graph(struct ggml_tensor* control_img,
                                    struct ggml_tensor* img,
                                    struct ggml_tensor* txt,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* pe,
                                    int control_type = -1) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, CONTROL_DIT_GRAPH_SIZE, false);
        
        control_img = to_backend(control_img);
        img = to_backend(img);
        txt = to_backend(txt);
        timesteps = to_backend(timesteps);
        if (pe != nullptr) {
            pe = to_backend(pe);
        }
        
        // Convert task_embedding to backend for use in forward
        struct ggml_tensor* task_emb_backend = nullptr;
        if (task_embedding != nullptr && control_type >= 0) {
            task_emb_backend = to_backend(task_embedding);
        }
        
        control_img = process_latent(compute_ctx, control_img);
        img = process_latent(compute_ctx, img);
        
        GGMLRunnerContext runner_ctx;
        runner_ctx.ggml_ctx = compute_ctx;
        runner_ctx.backend = runtime_backend;
        
        auto outs = control_net.forward(&runner_ctx, control_img, img, txt, timesteps, pe, 
                                         task_emb_backend, control_type);
        
        if (control_ctx == nullptr) {
            alloc_control_ctx(outs);
        }
        
        for (size_t i = 0; i < outs.size(); i++) {
            ggml_build_forward_expand(gf, ggml_cpy(compute_ctx, outs[i], controls[i]));
        }
        
        return gf;
    }
    
    void compute(int n_threads,
                 struct ggml_tensor* control_img,
                 struct ggml_tensor* img,
                 struct ggml_tensor* txt,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* pe,
                 int control_type = -1,
                 struct ggml_tensor** output = nullptr,
                 struct ggml_context* output_ctx = nullptr) {
        // Use provided control_type or fall back to current_control_type
        int type_to_use = (control_type >= 0) ? control_type : current_control_type;
        
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(control_img, img, txt, timesteps, pe, type_to_use);
        };
        
        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
    
    // Convenience overload with sd_control_type_t enum
    void compute(int n_threads,
                 struct ggml_tensor* control_img,
                 struct ggml_tensor* img,
                 struct ggml_tensor* txt,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* pe,
                 sd_control_type_t control_type,
                 struct ggml_tensor** output = nullptr,
                 struct ggml_context* output_ctx = nullptr) {
        compute(n_threads, control_img, img, txt, timesteps, pe, 
                static_cast<int>(control_type), output, output_ctx);
    }
    
    bool load_from_file(const std::string& file_path, int n_threads) {
        LOG_INFO("loading control net dit from '%s'", file_path.c_str());
        alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        get_param_tensors(tensors, "");  // Use our get_param_tensors which includes task_embedding
        
        std::set<std::string> ignore_tensors;
        
        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init control net dit model loader from file failed: '%s'", file_path.c_str());
            return false;
        }
        
        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads);
        
        if (!success) {
            LOG_ERROR("load control net dit tensors from model loader failed");
            return false;
        }
        
        LOG_INFO("control net dit model loaded (supports %d control types)", num_control_types);
        return success;
    }
    
    // Accumulate control signals from multiple hints
    // This allows using multiple control images (e.g., pose + depth) with a single Union ControlNet
    void accumulate_controls(const std::vector<struct ggml_tensor*>& new_controls, float strength = 1.0f) {
        if (controls.empty()) {
            LOG_ERROR("Cannot accumulate controls - no initial controls computed");
            return;
        }
        
        if (new_controls.size() != controls.size()) {
            LOG_ERROR("Control size mismatch: expected %zu, got %zu", controls.size(), new_controls.size());
            return;
        }
        
        // Add new_controls to existing controls (weighted by strength)
        for (size_t i = 0; i < controls.size(); i++) {
            if (controls[i] != nullptr && new_controls[i] != nullptr) {
                // Note: This requires a compute context - actual accumulation
                // should be done in the build_graph with proper GGML ops
                // This function is a placeholder for the API
            }
        }
    }
};

} // namespace QwenControlNet

#endif // __CONTROL_DIT_HPP__
