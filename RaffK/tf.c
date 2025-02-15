#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define tf_validation 1
#define tf_enc_file_size 722883
#define tf_data_size_max 1000000
#define tf_safetensor_file_size 548105171
#define tf_safetensor_json_size 14283
#define tf_d_vocab 50257
#define tf_d_seq 1024
#define tf_d_model 768
#define tf_d_k 64
#define tf_n_heads 12
#define tf_n_layers 12
#define tf_rsqrt_d_k 0.125f

struct tf_decoder_item {
    uint32_t offset;
    uint32_t size;
};

struct tf_decoder {
    struct tf_decoder_item items[tf_d_vocab];
    char raw[tf_enc_file_size - tf_d_vocab * sizeof(struct tf_decoder_item)];
};

struct tf_parameters {
    struct {
        float* weight;
    } wte;
    struct {
        float* weight;
    } wpe;
    struct {
        struct {
            float* bias;
            float* weight;
        } ln_1;
        struct {
            struct {
                float* bias;
                float* weight;
            } c_attn;
            struct {
                float* bias;
                float* weight;
            } c_proj;
        } attn;
        struct {
            float* bias;
            float* weight;
        } ln_2;
        struct {
            struct {
                float* bias;
                float* weight;
            } c_fc;
            struct {
                float* bias;
                float* weight;
            } c_proj;
        } mlp;
    } h[12];
    struct {
        float* bias;
        float* weight;
    } ln_f;
};

struct tf_activations {
    struct {
        float out[tf_d_seq][tf_d_model];
    } embedding;
    struct {
        struct {
            float r_std[tf_d_seq];
            float mean[tf_d_seq];
            float out[tf_d_seq][tf_d_model];
        } ln_1;
        struct {
            struct {
                float out[tf_d_seq][3 * tf_d_model];
            } c_attn;
            struct {
                float out[tf_n_heads][tf_d_seq][tf_d_seq];
            } softmax;
            struct {
                float out[tf_d_seq][tf_d_model];
            } z;
            struct {
                float out[tf_d_seq][tf_d_model];
            } c_proj;
        } attn;
        struct {
            float out[tf_d_seq][tf_d_model];
        } res_1;
        struct {
            float r_std[tf_d_seq];
            float mean[tf_d_seq];
            float out[tf_d_seq][tf_d_model];
        } ln_2;
        struct {
            struct {
                float out[tf_d_seq][4 * tf_d_model];
            } c_fc;
            struct {
                float out[tf_d_seq][4 * tf_d_model];
            } gelu;
            struct {
                float out[tf_d_seq][tf_d_model];
            } c_proj;
        } mlp;
        struct {
            float out[tf_d_seq][tf_d_model];
        } res_2;
    } h[12];
    struct {
        float r_std[tf_d_seq];
        float mean[tf_d_seq];
        float out[tf_d_seq][tf_d_model];
    } ln_f;
    struct {
        float out[tf_d_seq][tf_d_vocab];
    } unembedding;
};

struct tf_gradients {
    struct {
        float weight[tf_d_vocab][tf_d_model];
    } wte;
    struct {
        float weight[tf_d_seq][tf_d_model];
    } wpe;
    struct {
        struct {
            float weight[tf_d_model];
            float bias[tf_d_model];
        } ln_1;
        struct {
            struct {
                float weight[tf_d_model][3 * tf_d_model];
                float bias[3 * tf_d_model];
            } c_attn;
            struct {
                float weight[tf_d_model][tf_d_model];
                float bias[tf_d_model];
            } c_proj;
        } attn;
        struct {
            float weight[tf_d_model];
            float bias[tf_d_model];
        } ln_2;
        struct {
            struct {
                float weight[tf_d_model][4 * tf_d_model];
                float bias[4 * tf_d_model];
            } c_fc;
            struct {
                float weight[4 * tf_d_model][tf_d_model];
                float bias[tf_d_model];
            } c_proj;
        } mlp;
    } h[12];
    struct {
        float weight[tf_d_model];
        float bias[tf_d_model];
    } ln_f;
};

struct tf_activations_back {
    struct {
        float out[tf_d_seq][tf_d_model];
    } embedding;
    struct {
        float out[tf_d_seq][tf_d_model];
    } ln_1;
    struct {
        struct {
            float out[tf_d_seq][3 * tf_d_model];
        } c_attn;
        struct {
            float out[tf_d_seq];
        } softmax;
        struct {
            float out[tf_d_seq][tf_d_model];
        } z;
    } attn;
    struct {
        float in_2[tf_d_seq][tf_d_model];
        float out[tf_d_seq][tf_d_model];
    } res_1;
    struct {
        float out[tf_d_seq][tf_d_model];
    } ln_2;
    struct {
        struct {
            float out[tf_d_seq][4 * tf_d_model];
        } c_fc;
        struct {
            float out[tf_d_seq][4 * tf_d_model];
        } gelu;
    } mlp;
    struct {
        float in_2[tf_d_seq][tf_d_model];
        float out[tf_d_seq][tf_d_model];
    } res_2;
    struct {
        float out[tf_d_seq][tf_d_model];
    } ln_f;
    struct {
        float out[tf_d_seq][tf_d_vocab];
    } unembedding;
};

struct tf_add {
    const float* in_1;
    const float* in_2;
    float* out;
    size_t count;
};

struct tf_fc {
    const float* weight; // (v: in_count, u: out_count)
    const float* bias; // (v: 1, u: out_count)
    const float* in; // (v: sample_count, u: in_count)
    float* out; // (v: sample_count, u: out_count)
    size_t in_count;
    size_t out_count;
    size_t sample_count;
};

struct tf_fc_back {
    const float* weight; // (v: in_count, u: out_count)
    const float* in; // (v: sample_count, u: in_count)
    const float* dL_dout; // (v: sample_count, u: out_count)
    float* dL_dweight; // (v: in_count, u: out_count)
    float* dL_dbias; // (v: 1, u: out_count)
    float* dL_din; // (v: sample_count, u: in_count)
    size_t in_count;
    size_t out_count;
    size_t sample_count;
};

struct tf_ln {
    const float* weight; // (v: 1, u: in_count)
    const float* bias; // (v: 1, u: in_count)
    const float* in; // (v: sample_count, u: in_count)
    float* r_std; // (v: sample_count, u: 1)
    float* mean; // (v: sample_count, u: 1)
    float* out; // (v: sample_count, u: in_count)
    size_t in_count;
    size_t sample_count;
};

struct tf_ln_back {
    const float* weight; // (v: 1, u: in_count)
    const float* in; // (v: sample_count, u: in_count)
    const float* r_std; // (v: sample_count, u: 1)
    const float* mean; // (v: sample_count, u: 1)
    const float* dL_dout; // (v: sample_count, u: in_count)
    float* dL_dweight; // (v: 1, u: in_count)
    float* dL_dbias; // (v: 1, u: in_count)
    float* dL_din; // (v: sample_count, u: in_count)
    size_t in_count;
    size_t sample_count;
};

struct tf_validation_time {
    double t_start;
    double t_last;
    double embedding;
    double ln_1;
    struct {
        double c_attn;
        double z;
        double c_proj;
    } attn;
    double res_1;
    double ln_2;
    struct {
        double c_fc;
        double gelu;
        double c_proj;
    } mlp;
    double res_2;
    double ln_f;
    double unembedding;
    double total;
};

static double tf_current_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void tf_validation_time(double* target, double* t_last) {
    if (tf_validation) {
        double t = tf_current_time();
        *target += t - *t_last;
        *t_last = t;
    }
}

static void tf_validation_sum(float* in, size_t in_count, double expected_sum) {
    if (tf_validation) {
        float* in_end = in + in_count;
        double sum = 0.0;
        for (; in != in_end; in++) {
            sum += (double)*in;
        }
        if (expected_sum != sum) {
            fprintf(stderr, "expected: %.24f (%a), got %.24f (%a)\n", expected_sum, expected_sum, sum, sum);
            abort();
        }
    }
}

static void tf_offset_size(const char* json_raw, const char* str, size_t* out_offset, size_t* out_size) {
    char temp[32];
    char* start = strstr(json_raw, str);
    start = strstr(start, "data_offsets") + 15;
    char* end = strstr(start, ",");
    memcpy(temp, start, end - start);
    temp[end - start] = 0;
    size_t offset = (size_t)atoi(start);
    start = end + 1;
    end = strstr(start, "]");
    memcpy(temp, start, end - start);
    temp[end - start] = 0;
    size_t offset_end = (size_t)atoi(temp);
    size_t size = offset_end - offset;
    *out_offset = offset;
    *out_size = size;
}

static void tf_add(const struct tf_add* add) {
    const float* in_1 = add->in_1;
    const float* in_2 = add->in_2;
    float* out = add->out;
    float* out_end = out + add->count;
    for (; out != out_end; out++, in_1++, in_2++) {
        *out = *in_1 + *in_2;
    }
}

static void tf_fc(const struct tf_fc* fc) {
    for (size_t i = 0; i < fc->sample_count; i++) {
        float* out = fc->out + i * fc->out_count;
        float* out_end = out + fc->out_count;
        float* out_reset = out;

        const float* weight = fc->weight;
        const float* weight_end = weight + fc->in_count * fc->out_count;

        const float* in = fc->in + i * fc->in_count;

        memcpy(out, fc->bias, fc->out_count * sizeof(float));

        while (true) {
            *out += *in * *weight;
            out++;
            weight++;
            if (out == out_end) {
                out = out_reset;
                in++;
                if (weight == weight_end) {
                    break;
                }
            }
        }
    }
}

static void tf_fc_back(const struct tf_fc_back* fc) {
    for (size_t i = 0; i < fc->sample_count; i++) {
        const float* dL_dout = fc->dL_dout + i * fc->out_count;
        const float* dL_dout_end = dL_dout + fc->out_count;
        const float* dL_dout_reset = dL_dout;

        float* dL_dbias = fc->dL_dbias;

        for (; dL_dout != dL_dout_end; dL_dbias++, dL_dout++) {
            *dL_dbias += *dL_dout;
        }

        dL_dout = dL_dout_reset;

        const float* in = fc->in + i * fc->in_count;
        const float* in_end = in + fc->in_count;
        float* dL_dweight = fc->dL_dweight;

        const float* weight = fc->weight;
        float* dL_din = fc->dL_din + i * fc->in_count;

        float dot = 0.0f;

        while (true) {
            *dL_dweight += *dL_dout * *in;
            dot += *dL_dout * *weight;
            dL_dweight++;
            dL_dout++;
            weight++;
            if (dL_dout == dL_dout_end) {
                dL_dout = dL_dout_reset;
                in++;
                *dL_din = dot;
                dot = 0.0f;
                dL_din++;
                if (in == in_end) {
                    break;
                }
            }
        }
    }
}

static void tf_ln(const struct tf_ln* ln) {
    for (size_t i = 0; i < ln->sample_count; i++) {
        const float* in = ln->in + i * ln->in_count;
        const float* in_end = in + ln->in_count;
        const float* in_reset = in;

        float total = 0.0f;

        for (; in != in_end; in++) {
            total += *in;
        }

        float mean = total / (float)ln->in_count;
        ln->mean[i] = mean;

        float total_diff_squared = 0.0f;

        for (in = in_reset; in != in_end; in++) {
            float diff = (*in - mean);
            total_diff_squared += diff * diff;
        }

        float variance = total_diff_squared / (float)ln->in_count;
        float r_std = 1.0f / sqrtf(variance + 1e-5f);
        ln->r_std[i] = r_std;

        const float* weight = ln->weight;
        const float* bias = ln->bias;

        float* out = ln->out + i * ln->in_count;

        for (in = in_reset; in != in_end; in++, weight++, bias++, out++) {
            float n = (*in - mean) * r_std;
            *out = n * *weight + *bias;
        }
    }
}

static void tf_ln_back(const struct tf_ln_back* ln) {
    // L = s(y), s: everything after layernorm
    // y = layernorm(weight, bias, x)
    // layernorm(weight, bias, x) = in_norm * weight + bias
    // in_norm = (x - mean) / var;
    // dL/dweight = dL/dy * dy/dweight
    // dy/dweight = in_norm
    // dL/dbias = dL/dy * dy/dbias
    // dy/dbias = 1
    // dL/din_norm = dL/dy * dy/din_norm
    // dy/din_norm = weight

    for (size_t i = 0; i < ln->sample_count; i++) {
        float r_std = ln->r_std[i];
        float mean = ln->mean[i];

        float mean_partial = 0.0f;
        float var_partial = 0.0f;

        const float* dL_dout = ln->dL_dout + i * ln->in_count;
        const float* dL_dout_end = dL_dout + ln->in_count;
        const float* weight = ln->weight;
        const float* in = ln->in + i * ln->in_count;

        const float* dL_dout_reset = dL_dout;
        const float* weight_reset = weight;
        const float* in_reset = in;

        float* dL_dweight = ln->dL_dweight;
        float* dL_dbias = ln->dL_dbias;

        for (; dL_dout != dL_dout_end; dL_dout++, weight++, in++, dL_dweight++, dL_dbias++) {
            float dL_din_norm = *dL_dout * *weight;
            float in_norm = (*in - mean) * r_std;

            *dL_dweight += *dL_dout * in_norm;
            *dL_dbias += *dL_dout;

            mean_partial += dL_din_norm;
            var_partial += dL_din_norm * in_norm;
        }

        mean_partial /= ln->in_count;
        var_partial /= ln->in_count;

        dL_dout = dL_dout_reset;
        weight = weight_reset;
        in = in_reset;

        float* dL_din = ln->dL_din + i * ln->in_count;

        for (; dL_dout != dL_dout_end; dL_dout++, weight++, in++, dL_din++) {
            float dL_din_norm = *dL_dout * *weight;
            float in_norm = (*in - mean) * r_std;

            float dL_din_ = 0.0f;
            dL_din_ += dL_din_norm;
            dL_din_ -= mean_partial;
            dL_din_ -= var_partial * in_norm;
            dL_din_ *= r_std;
            *dL_din = dL_din_;
        }
    }
}

static void tf_process(struct tf_parameters* parameters, struct tf_activations* activations, uint16_t* input, size_t input_size, uint16_t* out_token, bool is_training, struct tf_gradients* gradients, struct tf_activations_back* activations_back, uint16_t* expected) {
    struct tf_validation_time time = { 0 };
    time.t_start = tf_current_time();
    time.t_last = time.t_start;

    for (size_t i = 0; i < input_size; i++) {
        float* wte = parameters->wte.weight + input[i] * tf_d_model;
        float* wte_end = wte + tf_d_model;

        float* wpe = parameters->wpe.weight + i * tf_d_model;

        float* out = (float*)activations->embedding.out + i * tf_d_model;

        for (; wte != wte_end; wte++, wpe++, out++) {
            *out = *wte + *wpe;
        }
    }

    tf_validation_time(&time.embedding, &time.t_last);
    tf_validation_sum((float*)activations->embedding.out, input_size * tf_d_model, -0x1.e86f2c2adep+4);

    for (int layer_i = 0; layer_i < tf_n_layers; layer_i++) {
        tf_ln(&(struct tf_ln) {
            .weight = parameters->h[layer_i].ln_1.weight,
                .bias = parameters->h[layer_i].ln_1.bias,
                .in = layer_i == 0 ? (float*)activations->embedding.out : (float*)activations->h[layer_i - 1].res_2.out,
                .r_std = (float*)activations->h[layer_i].ln_1.r_std,
                .mean = (float*)activations->h[layer_i].ln_1.mean,
                .out = (float*)activations->h[layer_i].ln_1.out,
                .in_count = tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.ln_1, &time.t_last);
        if (layer_i == 0) {
            tf_validation_sum((float*)activations->h[layer_i].ln_1.out, input_size * tf_d_model, -0x1.4e34ee18da56ap+8);
        }

        tf_fc(&(struct tf_fc) {
            .weight = parameters->h[layer_i].attn.c_attn.weight,
                .bias = parameters->h[layer_i].attn.c_attn.bias,
                .in = (float*)activations->h[layer_i].ln_1.out,
                .out = (float*)activations->h[layer_i].attn.c_attn.out,
                .in_count = tf_d_model,
                .out_count = 3 * tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.attn.c_attn, &time.t_last);
        if (layer_i == 0) {
            tf_validation_sum((float*)activations->h[layer_i].attn.c_attn.out, input_size * 3 * tf_d_model, -0x1.9f967d2b7f151p+11);
        }

        memset(activations->h[layer_i].attn.z.out, 0, sizeof(activations->h[layer_i].attn.z.out));

        for (size_t head_i = 0; head_i < tf_n_heads; head_i++) {
            for (size_t q_i = 0; q_i < input_size; q_i++) {
                float* softmax_out = activations->h[layer_i].attn.softmax.out[head_i][q_i];
                float softmax_max = -INFINITY;

                for (size_t k_i = 0; k_i <= q_i; k_i++) {
                    float* q = (float*)activations->h[layer_i].attn.c_attn.out + q_i * 3 * tf_d_model + head_i * tf_d_k;
                    float* q_end = q + tf_d_k;

                    float* k = (float*)activations->h[layer_i].attn.c_attn.out + k_i * 3 * tf_d_model + tf_d_model + head_i * tf_d_k;

                    float dot = 0.0f;

                    for (; q != q_end; q++, k++) {
                        dot += *q * *k;
                    }

                    dot *= tf_rsqrt_d_k;

                    softmax_out[k_i] = dot;

                    if (dot > softmax_max) {
                        softmax_max = dot;
                    }
                }

                float softmax_sum = 0.0f;

                for (size_t k_i = 0; k_i <= q_i; k_i++) {
                    float softmax_exp = expf(softmax_out[k_i] - softmax_max);
                    softmax_sum += softmax_exp;
                    softmax_out[k_i] = softmax_exp;
                }

                float r_softmax_sum = 1.0f / softmax_sum;

                for (size_t k_i = 0; k_i <= q_i; k_i++) {
                    softmax_out[k_i] *= r_softmax_sum;
                }

                for (size_t v_i = 0; v_i <= q_i; v_i++) {
                    float* v = (float*)activations->h[layer_i].attn.c_attn.out + v_i * 3 * tf_d_model + 2 * tf_d_model + head_i * tf_d_k;
                    float* v_end = v + tf_d_k;

                    float* z = (float*)activations->h[layer_i].attn.z.out + q_i * tf_d_model + head_i * tf_d_k;

                    float softmax_i = softmax_out[v_i];

                    for (; v != v_end; v++, z++) {
                        *z += softmax_i * *v;
                    }
                }
            }
        }

        tf_validation_time(&time.attn.z, &time.t_last);
        if (layer_i == 0) {
            tf_validation_sum((float*)activations->h[layer_i].attn.z.out, input_size * tf_d_model, 0x1.c64a4db1bfcdep+8);
        }

        tf_fc(&(struct tf_fc) {
            .weight = parameters->h[layer_i].attn.c_proj.weight,
                .bias = parameters->h[layer_i].attn.c_proj.bias,
                .in = (float*)activations->h[layer_i].attn.z.out,
                .out = (float*)activations->h[layer_i].attn.c_proj.out,
                .in_count = tf_d_model,
                .out_count = tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.attn.c_proj, &time.t_last);
        if (layer_i == 0) {
            tf_validation_sum((float*)activations->h[layer_i].attn.c_proj.out, input_size * tf_d_model, 0x1.850b3ffab297bp+8);
        }

        tf_add(&(struct tf_add) {
            .in_1 = layer_i == 0 ? (float*)activations->embedding.out : (float*)activations->h[layer_i - 1].res_2.out,
                .in_2 = (float*)activations->h[layer_i].attn.c_proj.out,
                .out = (float*)activations->h[layer_i].res_1.out,
                .count = input_size * tf_d_model,
        });

        tf_validation_time(&time.res_1, &time.t_last);

        tf_ln(&(struct tf_ln) {
            .weight = parameters->h[layer_i].ln_2.weight,
                .bias = parameters->h[layer_i].ln_2.bias,
                .in = (float*)activations->h[layer_i].res_1.out,
                .r_std = (float*)activations->h[layer_i].ln_2.r_std,
                .mean = (float*)activations->h[layer_i].ln_2.mean,
                .out = (float*)activations->h[layer_i].ln_2.out,
                .in_count = tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.ln_2, &time.t_last);
        if (layer_i == 0) {
            tf_validation_sum((float*)activations->h[layer_i].ln_2.out, input_size * tf_d_model, 0x1.188ffb5000f3dp+8);
        }

        tf_fc(&(struct tf_fc) {
            .weight = parameters->h[layer_i].mlp.c_fc.weight,
                .bias = parameters->h[layer_i].mlp.c_fc.bias,
                .in = (float*)activations->h[layer_i].ln_2.out,
                .out = (float*)activations->h[layer_i].mlp.c_fc.out,
                .in_count = tf_d_model,
                .out_count = 4 * tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.mlp.c_fc, &time.t_last);

        {
            const float* in = (float*)activations->h[layer_i].mlp.c_fc.out;
            const float* in_end = in + input_size * 4 * tf_d_model;
            float* out = (float*)activations->h[layer_i].mlp.gelu.out;
            for (; in != in_end; in++, out++) {
                float phi = 0.5f * (1.0f + erff(*in * (float)M_SQRT1_2));
                *out = *in * phi;
            }
        }

        tf_validation_time(&time.mlp.gelu, &time.t_last);

        tf_fc(&(struct tf_fc) {
            .weight = parameters->h[layer_i].mlp.c_proj.weight,
                .bias = parameters->h[layer_i].mlp.c_proj.bias,
                .in = (float*)activations->h[layer_i].mlp.gelu.out,
                .out = (float*)activations->h[layer_i].mlp.c_proj.out,
                .in_count = 4 * tf_d_model,
                .out_count = tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.mlp.c_proj, &time.t_last);
        if (layer_i == 0) {
            tf_validation_sum((float*)activations->h[layer_i].mlp.c_proj.out, input_size * tf_d_model, -0x1.012ce31d82fb8p+9);
        }

        tf_add(&(struct tf_add) {
            .in_1 = (float*)activations->h[layer_i].res_1.out,
                .in_2 = (float*)activations->h[layer_i].mlp.c_proj.out,
                .out = (float*)activations->h[layer_i].res_2.out,
                .count = input_size * tf_d_model,
        });

        tf_validation_time(&time.res_2, &time.t_last);
    }

    tf_ln(&(struct tf_ln) {
        .weight = parameters->ln_f.weight,
            .bias = parameters->ln_f.bias,
            .in = (float*)activations->h[11].res_2.out,
            .r_std = (float*)activations->ln_f.r_std,
            .mean = (float*)activations->ln_f.mean,
            .out = (float*)activations->ln_f.out,
            .in_count = tf_d_model,
            .sample_count = input_size,
    });

    tf_validation_time(&time.ln_f, &time.t_last);
    tf_validation_sum((float*)activations->ln_f.out, input_size * tf_d_model, 0x1.0437f5b8f47d8p+14);

    for (size_t i = is_training ? 0 : input_size - 1; i < input_size; i++) {
        float* out = activations->unembedding.out[i];
        float* out_end = out + tf_d_vocab;
        float* out_reset = out;

        const float* weight = parameters->wte.weight;
        const float* weight_end = weight + tf_d_vocab * tf_d_model;

        const float* in = (float*)activations->ln_f.out[i];
        const float* in_end = in + tf_d_model;
        const float* in_reset = in;

        float dot = 0.0f;
        float softmax_max = -INFINITY;

        while (true) {
            dot += *weight * *in;
            weight++;
            in++;
            if (in == in_end) {
                in = in_reset;
                *out = dot;
                out++;
                if (dot > softmax_max) {
                    softmax_max = dot;
                }
                dot = 0.0f;
                if (weight == weight_end) {
                    break;
                }
            }
        }

        float softmax_exp_sum = 0.0f;

        for (out = out_reset; out != out_end; out++) {
            float softmax_exp_i = expf(*out - softmax_max);
            softmax_exp_sum += softmax_exp_i;
            *out = softmax_exp_i;
        }

        float r_softmax_exp_sum = 1.0f / softmax_exp_sum;

        for (out = out_reset; out != out_end; out++) {
            *out *= r_softmax_exp_sum;
        }
    }

    tf_validation_time(&time.unembedding, &time.t_last);
    tf_validation_sum((float*)activations->unembedding.out, input_size * tf_d_vocab, 0x1.0008be62ee50cp+6);

    tf_validation_time(&time.total, &time.t_start);
    if (tf_validation) {
        __builtin_dump_struct(&time, printf);
    }


    if (!is_training) {
        const float* out = activations->unembedding.out[input_size - 1];
        const float* out_end = out + tf_d_vocab;
        const float* out_max = out;

        for (; out != out_end; out++) {
            if (*out > *out_max) {
                out_max = out;
            }
        }

        *out_token = out_max - activations->unembedding.out[input_size - 1];
        return;
    }

    memset(&time, 0, sizeof(time));
    time.t_start = tf_current_time();
    time.t_last = time.t_start;

    // L = crossentropy(softmax(x))
    // dL/dx = p - 1, for the token that was expected
    // dL/dx = p, for the token that was not expected
    // x is the result of unembedding with wte

    memcpy(activations_back->unembedding.out, activations->unembedding.out, input_size * tf_d_vocab * sizeof(float));

    float loss_total = 0.0f;

    for (size_t i = 0; i < input_size; i++) {
        uint16_t correct = expected[i];
        float p_correct = activations_back->unembedding.out[i][correct];
        activations_back->unembedding.out[i][correct] = p_correct - 1.0f;

        float cross_entropy_loss = -logf(p_correct);
        loss_total += cross_entropy_loss;
    }

    tf_validation_sum(&loss_total, 1, 0x1.08868ep+8);

    float r_input_size = 1.0f / (float)input_size;

    memset(activations_back->ln_f.out, 0, sizeof(activations_back->ln_f.out));

    for (size_t i = 0; i < input_size; i++) {
        float* dL_din = (float*)activations_back->ln_f.out[i];
        float* dL_din_end = dL_din + tf_d_model;
        float* dL_din_reset = dL_din;
        const float* in = activations->ln_f.out[i];
        const float* in_end = in + tf_d_model;
        const float* in_reset = in;

        const float* dL_dout = activations_back->unembedding.out[i];
        const float* dL_dout_end = dL_dout + tf_d_vocab;

        float* dL_dweight = (float*)gradients->wte.weight;
        const float* weight = parameters->wte.weight;

        while (true) {
            *dL_dweight += *dL_dout * *in * r_input_size;
            *dL_din += *dL_dout * *weight;
            dL_din++;
            in++;
            dL_dweight++;
            weight++;
            if (in == in_end) {
                in = in_reset;
                dL_din = dL_din_reset;
                dL_dout++;
                if (dL_dout == dL_dout_end) {
                    break;
                }
            }
        }

        for (; dL_din != dL_din_end; dL_din++) {
            *dL_din *= r_input_size;
        }
    }

    tf_validation_time(&time.unembedding, &time.t_last);
    tf_validation_sum((float*)gradients->wte.weight, tf_d_vocab * tf_d_model, 0x1.7f5c4d133539p-6);
    tf_validation_sum((float*)activations_back->ln_f.out, input_size * tf_d_model, -0x1.4adb0746d50fap-5);

    tf_ln_back(&(struct tf_ln_back) {
        .weight = parameters->ln_f.weight,
            .in = (float*)activations->h[11].res_2.out,
            .r_std = (float*)activations->ln_f.r_std,
            .mean = (float*)activations->ln_f.mean,
            .dL_dout = (float*)activations_back->ln_f.out,
            .dL_dweight = (float*)gradients->ln_f.weight,
            .dL_dbias = (float*)gradients->ln_f.bias,
            .dL_din = (float*)activations_back->res_2.out,
            .in_count = tf_d_model,
            .sample_count = input_size,
    });

    tf_validation_time(&time.ln_f, &time.t_last);
    tf_validation_sum((float*)gradients->ln_f.weight, tf_d_model, -0x1.1264bf0ff8p-1);
    tf_validation_sum((float*)gradients->ln_f.bias, tf_d_model, -0x1.4adaf7184p-5);
    tf_validation_sum((float*)activations_back->res_2.out, input_size * tf_d_model, -0x1.690e462bp-27);

    for (int layer_i = tf_n_layers - 1; layer_i >= 0; layer_i--) {
        tf_fc_back(&(struct tf_fc_back) {
            .weight = (float*)parameters->h[layer_i].mlp.c_proj.weight,
                .in = (float*)activations->h[layer_i].mlp.gelu.out,
                .dL_dout = (float*)activations_back->res_2.out,
                .dL_dweight = (float*)gradients->h[layer_i].mlp.c_proj.weight,
                .dL_dbias = gradients->h[layer_i].mlp.c_proj.bias,
                .dL_din = (float*)activations_back->mlp.gelu.out,
                .in_count = 4 * tf_d_model,
                .out_count = tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.mlp.c_proj, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)gradients->h[layer_i].mlp.c_proj.weight, 4 * tf_d_model * tf_d_model, -0x1.3d7e7fe8p-23);
            tf_validation_sum((float*)gradients->h[layer_i].mlp.c_proj.bias, tf_d_model, -0x1.6264p-27);
            tf_validation_sum((float*)activations_back->mlp.gelu.out, input_size * 4 * tf_d_model, -0x1.627d327e64f6cp-1);
        }

        {
            const float* in = (float*)activations->h[layer_i].mlp.c_fc.out;
            const float* dL_dout = (float*)activations_back->mlp.gelu.out;
            float* dL_din = (float*)activations_back->mlp.c_fc.out;
            float* dL_din_end = dL_din + input_size * 4 * tf_d_model;

            for (; dL_din != dL_din_end; in++, dL_dout++, dL_din++) {
                float phi = 0.5f * (1.0f + erff(*in * (float)M_SQRT1_2));
                float dgelu_din = phi + *in * (1.0f / sqrtf(2.0f * (float)M_PI)) * expf(-0.5f * *in * *in);
                *dL_din = *dL_dout * dgelu_din;
            }
        }

        tf_validation_time(&time.mlp.gelu, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)activations_back->mlp.c_fc.out, input_size * 4 * tf_d_model, 0x1.2e4527c197bcbp-5);
        }

        tf_fc_back(&(struct tf_fc_back) {
            .weight = (float*)parameters->h[layer_i].mlp.c_fc.weight,
                .in = (float*)activations->h[layer_i].ln_2.out,
                .dL_dout = (float*)activations_back->mlp.c_fc.out,
                .dL_dweight = (float*)gradients->h[layer_i].mlp.c_fc.weight,
                .dL_dbias = gradients->h[layer_i].mlp.c_fc.bias,
                .dL_din = (float*)activations_back->ln_2.out,
                .in_count = tf_d_model,
                .out_count = 4 * tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.mlp.c_fc, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)gradients->h[layer_i].mlp.c_fc.weight, tf_d_model * 4 * tf_d_model, -0x1.200760d8e8494p+1);
            tf_validation_sum((float*)gradients->h[layer_i].mlp.c_fc.bias, 4 * tf_d_model, 0x1.2e452d5568p-5);
            tf_validation_sum((float*)activations_back->ln_2.out, input_size * tf_d_model, 0x1.e762c70de442dp+0);
        }

        tf_ln_back(&(struct tf_ln_back) {
            .weight = parameters->h[layer_i].ln_2.weight,
                .in = (float*)activations->h[layer_i].res_1.out,
                .r_std = (float*)activations->h[layer_i].ln_2.r_std,
                .mean = (float*)activations->h[layer_i].ln_2.mean,
                .dL_dout = (float*)activations_back->ln_2.out,
                .dL_dweight = (float*)gradients->h[layer_i].ln_2.weight,
                .dL_dbias = (float*)gradients->h[layer_i].ln_2.bias,
                .dL_din = (float*)activations_back->res_2.in_2,
                .in_count = tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.ln_2, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)gradients->h[layer_i].ln_2.weight, tf_d_model, -0x1.f2a6e6d9a1p-1);
            tf_validation_sum((float*)gradients->h[layer_i].ln_2.bias, tf_d_model, 0x1.e762c5ac36p+0);
            tf_validation_sum((float*)activations_back->res_2.in_2, input_size * tf_d_model, -0x1.3869712ap-27);
        }

        tf_add(&(struct tf_add) {
            .in_1 = (float*)activations_back->res_2.out,
                .in_2 = (float*)activations_back->res_2.in_2,
                .out = (float*)activations_back->res_1.out,
                .count = input_size * tf_d_model,
        });

        tf_validation_time(&time.res_2, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)activations_back->res_1.out, input_size * tf_d_model, -0x1.50b057dap-26);
        }

        tf_fc_back(&(struct tf_fc_back) {
            .weight = (float*)parameters->h[layer_i].attn.c_proj.weight,
                .in = (float*)activations->h[layer_i].attn.z.out,
                .dL_dout = (float*)activations_back->res_1.out,
                .dL_dweight = (float*)gradients->h[layer_i].attn.c_proj.weight,
                .dL_dbias = gradients->h[layer_i].attn.c_proj.bias,
                .dL_din = (float*)activations_back->attn.z.out,
                .in_count = tf_d_model,
                .out_count = tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.attn.c_proj, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)gradients->h[layer_i].attn.c_proj.weight, tf_d_model * tf_d_model, -0x1.f4f260d4p-24);
            tf_validation_sum((float*)gradients->h[layer_i].attn.c_proj.bias, tf_d_model, -0x1.564ep-26);
            tf_validation_sum((float*)activations_back->attn.z.out, input_size * tf_d_model, -0x1.0780431d99d0fp-3);
        }

        memset(activations_back->attn.c_attn.out, 0, sizeof(activations_back->attn.c_attn.out));

        for (size_t h = 0; h < tf_n_heads; h++) {
            for (size_t i = 0; i < input_size; i++) {
                const float* dL_dout = activations_back->attn.z.out[i] + h * tf_d_k;
                const float* dL_dout_end = dL_dout + tf_d_k;
                const float* dL_dout_reset = dL_dout;

                float* dL_dv = (float*)activations_back->attn.c_attn.out + 2 * tf_d_model + h * tf_d_k;
                const float* v = (float*)activations->h[layer_i].attn.c_attn.out + 2 * tf_d_model + h * tf_d_k;

                float* dL_da = (float*)activations_back->attn.softmax.out;
                float* dL_da_reset = dL_da;
                const float* a = (float*)activations->h[layer_i].attn.softmax.out[h][i];
                const float* a_i_end = a + i + 1;
                const float* a_end = a + tf_d_seq;
                const float* a_reset = a;

                float row_sum = 0.0f;
                float dot = 0.0f;

                while (true) {
                    if (a < a_i_end) {
                        *dL_dv += *dL_dout * *a;
                    }
                    dot += *dL_dout * *v;
                    dL_dout++;
                    dL_dv++;
                    v++;
                    if (dL_dout == dL_dout_end) {
                        dL_dout = dL_dout_reset;
                        dL_dv += 3 * tf_d_model - tf_d_k;
                        v += 3 * tf_d_model - tf_d_k;
                        if (a < a_i_end) {
                            row_sum += *a * dot;
                        }
                        *dL_da = dot;
                        dot = 0.0f;
                        dL_da++;
                        a++;
                        if (a == a_end) {
                            break;
                        }
                    }
                }

                dL_da = dL_da_reset;
                a = a_reset;

                float* dL_dq = (float*)activations_back->attn.c_attn.out[i] + h * tf_d_k;
                float* dL_dq_end = dL_dq + tf_d_k;
                float* dL_dq_reset = dL_dq;
                const float* q = (float*)activations->h[layer_i].attn.c_attn.out[i] + h * tf_d_k;
                const float* q_reset = q;

                float* dL_dk = (float*)activations_back->attn.c_attn.out + tf_d_model + h * tf_d_k;
                const float* k = (float*)activations->h[layer_i].attn.c_attn.out + tf_d_model + h * tf_d_k;

                while (true) {
                    float dL_ds = *a * (*dL_da - row_sum);
                    *dL_dq += dL_ds * *k * tf_rsqrt_d_k;
                    *dL_dk += dL_ds * *q * tf_rsqrt_d_k;
                    dL_dq++;
                    q++;
                    dL_dk++;
                    k++;
                    if (dL_dq == dL_dq_end) {
                        dL_dq = dL_dq_reset;
                        q = q_reset;
                        dL_dk += 3 * tf_d_model - tf_d_k;
                        k += 3 * tf_d_model - tf_d_k;
                        a++;
                        dL_da++;
                        if (a == a_i_end) {
                            break;
                        }
                    }
                }
            }
        }

        tf_validation_time(&time.attn.z, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)activations_back->attn.c_attn.out, input_size * 3 * tf_d_model, -0x1.1fc8ca5c4b7b6p-3);
        }

        tf_fc_back(&(struct tf_fc_back) {
            .weight = (float*)parameters->h[layer_i].attn.c_attn.weight,
                .in = (float*)activations->h[layer_i].ln_1.out,
                .dL_dout = (float*)activations_back->attn.c_attn.out,
                .dL_dweight = (float*)gradients->h[layer_i].attn.c_attn.weight,
                .dL_dbias = gradients->h[layer_i].attn.c_attn.bias,
                .dL_din = (float*)activations_back->ln_1.out,
                .in_count = tf_d_model,
                .out_count = 3 * tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.attn.c_attn, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)gradients->h[layer_i].attn.c_attn.weight, tf_d_model * 3 * tf_d_model, -0x1.689f62d6a3abdp+0);
            tf_validation_sum((float*)gradients->h[layer_i].attn.c_attn.bias, 3 * tf_d_model, -0x1.1fc8c86b866a4p-3);
            tf_validation_sum((float*)activations_back->ln_1.out, input_size * tf_d_model, 0x1.96e7bad126e79p-1);
        }

        tf_ln_back(&(struct tf_ln_back) {
            .weight = parameters->h[layer_i].ln_1.weight,
                .in = layer_i == 0 ? (float*)activations->embedding.out : (float*)activations->h[layer_i - 1].res_2.out,
                .r_std = (float*)activations->h[layer_i].ln_1.r_std,
                .mean = (float*)activations->h[layer_i].ln_1.mean,
                .dL_dout = (float*)activations_back->ln_1.out,
                .dL_dweight = (float*)gradients->h[layer_i].ln_1.weight,
                .dL_dbias = (float*)gradients->h[layer_i].ln_1.bias,
                .dL_din = (float*)activations_back->res_1.in_2,
                .in_count = tf_d_model,
                .sample_count = input_size,
        });

        tf_validation_time(&time.ln_1, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)gradients->h[layer_i].ln_1.weight, tf_d_model, -0x1.760ab5b77p-2);
            tf_validation_sum((float*)gradients->h[layer_i].ln_1.bias, tf_d_model, 0x1.96e7bdf08ap-1);
            tf_validation_sum((float*)activations_back->res_1.in_2, input_size * tf_d_model, 0x1.6d00ac5p-27);
        }

        tf_add(&(struct tf_add) {
            .in_1 = (float*)activations_back->res_1.out,
                .in_2 = (float*)activations_back->res_1.in_2,
                .out = layer_i == 0 ? (float*)activations_back->embedding.out : (float*)activations_back->res_2.out,
                .count = input_size * tf_d_model,
        });

        tf_validation_time(&time.res_1, &time.t_last);
        if (layer_i == 11) {
            tf_validation_sum((float*)activations_back->res_2.out, input_size * tf_d_model, -0x1.49a7d8p-27);
        }
    }

    for (size_t i = 0; i < input_size; i++) {
        const float* dL_dout = (float*)activations_back->embedding.out + i * tf_d_model;
        const float* dL_dout_end = dL_dout + tf_d_model;
        float* dL_dwte_weight = (float*)gradients->wte.weight + input[i] * tf_d_model;
        float* dL_dwpe_weight = (float*)gradients->wpe.weight + i * tf_d_model;

        for (; dL_dout != dL_dout_end; dL_dout++, dL_dwte_weight++, dL_dwpe_weight++) {
            *dL_dwte_weight += *dL_dout;
            *dL_dwpe_weight += *dL_dout;
        }
    }

    tf_validation_time(&time.embedding, &time.t_last);
    tf_validation_sum((float*)gradients->wte.weight, tf_d_vocab * tf_d_model, 0x1.7f75a7f7bb39p-6);
    tf_validation_sum((float*)gradients->wpe.weight, input_size * tf_d_model, 0x1.e96f7cp-20);

    tf_validation_time(&time.total, &time.t_start);
    if (tf_validation) {
        __builtin_dump_struct(&time, printf);
    }
}

int main(void) {
#define tf_align(x) (((x) + 255) & ~(size_t)0xff)
    size_t total_offset = 0;
    size_t decoder_offset = total_offset;
    total_offset += tf_align(sizeof(struct tf_decoder));
    size_t data_offset = total_offset;
    total_offset += tf_align(tf_data_size_max);
    size_t json_raw_offset = total_offset;
    total_offset += tf_align(tf_safetensor_json_size);
    size_t parameters_raw_offset = total_offset;
    total_offset += tf_align(tf_safetensor_file_size);
    size_t parameters_offset = total_offset;
    total_offset += tf_align(sizeof(struct tf_parameters));
    size_t activations_offset = total_offset;
    total_offset += tf_align(sizeof(struct tf_activations));
    size_t gradients_offset = total_offset;
    total_offset += tf_align(sizeof(struct tf_gradients));
    size_t activations_back_offset = total_offset;
    total_offset += tf_align(sizeof(struct tf_activations_back));
#undef tf_align

    fprintf(stderr, "requires %lu mib\n", total_offset >> 20);

    char* tf_raw = aligned_alloc(256, total_offset);
    assert(tf_raw && "allocation failed");

    struct tf_decoder* decoder = (void*)(tf_raw + decoder_offset);
    uint16_t* data = (void*)(tf_raw + data_offset);
    char* json_raw = (void*)(tf_raw + json_raw_offset);
    char* parameters_raw = (void*)(tf_raw + parameters_raw_offset);
    struct tf_parameters* parameters = (void*)(tf_raw + parameters_offset);
    struct tf_activations* activations = (void*)(tf_raw + activations_offset);
    struct tf_gradients* gradients = (void*)(tf_raw + gradients_offset);
    struct tf_activations_back* activations_back = (void*)(tf_raw + activations_back_offset);

    char* base_path = ".";
    size_t data_token_count;

    {
        char enc_path[PATH_MAX];
        snprintf(enc_path, sizeof(enc_path), "%s/assets/enc", base_path);

        FILE* f = fopen(enc_path, "r");
        assert(f);

        struct stat s;
        fstat(fileno(f), &s);
        assert(s.st_size == tf_enc_file_size);

        size_t read_size = tf_d_vocab * sizeof(struct tf_decoder_item);
        size_t read = fread(&decoder->items, 1, read_size, f);
        assert(read == read_size);

        read_size = tf_enc_file_size - read_size;
        read = fread(&decoder->raw, 1, read_size, f);
        assert(read == read_size);

        fclose(f);

        for (int i = 0; i < tf_d_vocab; i++) {
            // printf("%.5d: %.*s\n", i, decoder->items[i].size, decoder->raw + decoder->items[i].offset);
            assert(decoder->items[i].size != 0);
        }
        assert(decoder->items[tf_d_vocab - 1].offset + decoder->items[tf_d_vocab - 1].size == sizeof(decoder->raw));
    }
    {
        char data_path[PATH_MAX];
        snprintf(data_path, sizeof(data_path), "%s/assets/data", base_path);

        FILE* f = fopen(data_path, "r");
        assert(f);

        struct stat s;
        fstat(fileno(f), &s);
        assert(s.st_size <= tf_data_size_max);
        assert((size_t)s.st_size % sizeof(uint16_t) == 0);

        data_token_count = (size_t)s.st_size / sizeof(uint16_t);

        size_t read = fread(data, 1, (size_t)s.st_size, f);
        assert(read == (size_t)s.st_size);

        fclose(f);

        for (size_t i = 0; i < data_token_count && false; i++) {
            uint16_t token = data[i];
            write(STDOUT_FILENO, decoder->raw + decoder->items[token].offset, decoder->items[token].size);
        }
    }
    {
        char parameters_path[PATH_MAX];
        snprintf(parameters_path, sizeof(parameters_path), "%s/assets/model.safetensors", base_path);
        // https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors

        FILE* f = fopen(parameters_path, "r");
        assert(f);

        struct stat s;
        fstat(fileno(f), &s);
        assert((size_t)s.st_size == tf_safetensor_file_size);

        uint64_t json_size;
        size_t read = fread(&json_size, 1, sizeof(json_size), f);
        assert(read == sizeof(json_size));
        assert(json_size == tf_safetensor_json_size);

        read = fread(json_raw, 1, json_size, f);
        assert(read == json_size);

        long pos = ftell(f);
        assert(pos != -1);

        size_t raw_size = tf_safetensor_file_size - (size_t)pos;
        read = fread(parameters_raw, 1, raw_size, f);
        assert(read == raw_size);

        size_t offset;
        size_t size;
        tf_offset_size(json_raw, "wte", &offset, &size);
        parameters->wte.weight = (float*)(parameters_raw + offset);
        assert(size == tf_d_vocab * tf_d_model * sizeof(float));

        tf_offset_size(json_raw, "wpe", &offset, &size);
        parameters->wpe.weight = (float*)(parameters_raw + offset);
        assert(size == tf_d_seq * tf_d_model * sizeof(float));

        for (int h = 0; h < 12; h++) {
            char label[32];

            snprintf(label, sizeof(label), "h.%d.ln_1.bias", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].ln_1.bias = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.ln_1.weight", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].ln_1.weight = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.attn.c_attn.bias", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].attn.c_attn.bias = (float*)(parameters_raw + offset);
            assert(size == 3 * tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.attn.c_attn.weight", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].attn.c_attn.weight = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * 3 * tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.attn.c_proj.bias", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].attn.c_proj.bias = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.attn.c_proj.weight", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].attn.c_proj.weight = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.ln_2.bias", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].ln_2.bias = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.ln_2.weight", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].ln_2.weight = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.mlp.c_fc.bias", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].mlp.c_fc.bias = (float*)(parameters_raw + offset);
            assert(size == 4 * tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.mlp.c_fc.weight", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].mlp.c_fc.weight = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * 4 * tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.mlp.c_proj.bias", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].mlp.c_proj.bias = (float*)(parameters_raw + offset);
            assert(size == tf_d_model * sizeof(float));

            snprintf(label, sizeof(label), "h.%d.mlp.c_proj.weight", h);
            tf_offset_size(json_raw, label, &offset, &size);
            parameters->h[h].mlp.c_proj.weight = (float*)(parameters_raw + offset);
            assert(size == 4 * tf_d_model * tf_d_model * sizeof(float));
        }

        tf_offset_size(json_raw, "ln_f.bias", &offset, &size);
        parameters->ln_f.bias = (float*)(parameters_raw + offset);
        assert(size == tf_d_model * sizeof(float));

        tf_offset_size(json_raw, "ln_f.weight", &offset, &size);
        parameters->ln_f.weight = (float*)(parameters_raw + offset);
        assert(size == tf_d_model * sizeof(float));

        fclose(f);
    }

    bool is_training = tf_validation;

    if (is_training) {
        size_t input_size = 64;
        uint16_t* input = data;
        uint16_t* expected = input + 1;

        memset(gradients, 0, sizeof(*gradients));

        if (tf_validation) {
            arc4random_buf(activations, sizeof(*activations));
            arc4random_buf(activations_back, sizeof(*activations_back));
        }

        tf_process(parameters, activations, input, input_size, NULL, true, gradients, activations_back, expected);
    }
    else {
        size_t input_size = data_token_count < 64 ? data_token_count : 64;
        uint16_t* input = data;

        for (size_t i = 0; i < input_size; i++) {
            write(STDOUT_FILENO, decoder->raw + decoder->items[input[i]].offset, decoder->items[input[i]].size);
        }

        for (int i = 0; i < 128; i++) {
            uint16_t out_token;
            tf_process(parameters, activations, input, input_size, &out_token, false, NULL, NULL, NULL);
            write(STDOUT_FILENO, decoder->raw + decoder->items[out_token].offset, decoder->items[out_token].size);
            input[input_size++] = out_token;
        }
    }

    free(tf_raw);
}

// notes:
// matmul of (v1, u1) x (v2, u2) -> (v1, u2); u1 and v2 must match, output: (v1, u2)
// dot product of row of matrix 1 with column of matrix 2 produces scalar: (1, 4) x (4, 1) -> (1, 1)
// column of matrix 1 with row of matrix 2 produces matrix: (4, 1) x (1, 4) -> (4, 4)
// add up all resulting matrices for final result
// by default, numpy array are laid out row-major, same for C (arr[v][u], arr[row][col], u changes most often)
// architecture: d_seq = 1024, d_model = 768, n_heads = 12, n_layers = 12, d_k = d_model / n_heads = 64
// input processing:
// text fragment to token: process text with tokenizer. currently done in python using openai tiktoken, so were just loading
// the encoded text as a sequence of u16.
// token embedding: tranform the token to a vector. formally, you could call this a transform from a d_vocab-dim space
// to a d_model-dim space ((v: d_model, u: d_vocab) x (v: d_vocab, u: 1) -> (v: d_model, u: 1)). in reality, only one element of
// the input would be 1, the rest is zero, so its equivalent to simply selecting a column from the transformation matrix
// transpose result and repeat for each element in the sequence: (v: input_size, u: d_model)
// position embedding: create a vector for each element in the sequence encoding its position (v: input_size, u: d_model)
// token encoding and position encoding are simply added (v: input_size, u: d_model)
// alternating attention and mlp processing:
// 12 layers of (normalization, attention, residual, normalization, mlp, residual):
// normalization: each element in the sequence gets transformed independently, using the same weight and bias.
// simple per-element multiplication of the d_model-dimensional vectors with weight and addition with bias
// norm(x) = (x - "mean of elements of the vector") / "stddev of elements of the vector, sqrt(var)") * weight + bias
// attention: get query, key, and value vectors via transform into lower-dimensional space: 768 -> 64
// (v: d_k, u: d_model) x (v: d_model, u: 1) -> (v: d_k, u: 1), in code: transposed (v: input_size, u: d_k)
// compare each key with all queries via dot product, divide by sqrt(d_k), softmax, weighted sum of value vectors
// for num stability, softmax first subtracts max value before exp(), equivalent because:
// exp(a - b) = exp(a) / exp(b), so:
// exp(x) / sum(exp(x))
// exp(x - max) / sum(exp(x - max))
// (exp(x) * exp(max)^-1) / (sum(exp(x)) * exp(max)^-1)
// transform resulting vectors from d_k dimensions back to d_model dimensions, implemented by concatenating and applying
// a single matmul attn.c_proj.weight + attn.c_proj.bias (v: d_model, u: d_k) x (v: d_k, u: 1) -> (v: d_model, u: 1)
// normalization again
// mlp: first (v: 4*d_model, u: d_model) x (v: d_model, u: 1) -> (v: 4*d_model, u: 1)
// then gelu (v: 4*d_model, u: 1) -> (v: 4*d_model, u: 1)
// then (v: d_model, u: 4*d_model) x (v: 4*d_model, u: 1) -> (v: d_model, u: 1)
// for better cache locality, these are all transposed in the code
// residual 1: x = x + attn(norm(x))
// residual 2: x = x + mlp(norm(x))
// output processing:
// normalization
// matmul with wte to get back to d_vocab dimensions
// (v: d_vocab, u: d_model) x (v: d_model, u: 1) -> (v: d_vocab, u: 1), choose output based on probability
// about negative log-likelihood vs cross entropy loss:
// ce: negative sum of (expected_p * log(actual_p))
// nll: -log(p_correct)
// for a one-hot distribution, they are equivalent
// in general, ce is used to compare two distributions
// for training, you would average the loss across the sequence and the batch
// when doing backprop through wte, we have many cache misses, but doing it more optimally
// in row-major order produces different results (tested with double to make sure its the order)