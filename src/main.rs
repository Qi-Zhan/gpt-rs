#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use std::{
    fs::File,
    io::{Read, Write},
};

#[cfg(not(feature = "noparallel"))]
use rayon::prelude::*;
use tiktoken_rs::r50k_base;

const N: usize = 10;
const GPT2_EOT: u32 = 50256;
const NUM_PARAMETER_TENSORS: usize = 16;
const NUM_ACTIVATION_TENSORS: usize = 23;

struct GPT2 {
    config: GPT2Config,
    /// the weights (parameters) of the model, and the optimizer
    params: ParameterTensors,
    /// the activations of the model
    acts: ActivationTensors,
    act_sizes: [usize; NUM_ACTIVATION_TENSORS],
    num_activations: usize,
    // other run state configuration
    /// the batch size (B) of current forward pass
    batch_size: usize,
    /// the sequence length (T) of current forward pass
    seq_len: usize,
}

impl GPT2 {
    fn build_from_checkpoint(checkpoint_path: &str) -> Self {
        // read in model from a checkpoint file
        let mut model_file = File::open(checkpoint_path).expect("Error opening model file");
        // read 256 int32
        let mut model_header: [u32; 256] = [0; 256];
        model_file
            .read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    model_header.as_mut_ptr() as *mut u8,
                    model_header.len() * std::mem::size_of::<i32>(),
                )
            })
            .expect("Error reading model file");
        if model_header[0] != 20240326 {
            panic!("Bad magic model file");
        }
        if model_header[1] != 1 {
            panic!("Bad version model file");
        }
        // read in hyperparameters
        let maxT = model_header[2] as usize;
        let V = model_header[3] as usize;
        let L = model_header[4] as usize;
        let NH = model_header[5] as usize;
        let C = model_header[6] as usize;

        let config = GPT2Config {
            vocab_size: V,
            num_layers: L,
            num_heads: NH,
            channels: C,
        };

        let param_sizes = [
            V * C,           // wte
            maxT * C,        // wpe
            L * C,           // ln1w
            L * C,           // ln1b
            L * (3 * C) * C, // qkvw
            L * (3 * C),     // qkvb
            L * C * C,       // attprojw
            L * C,           // attprojb
            L * C,           // ln2w
            L * C,           // ln2b
            L * (4 * C) * C, // fcw
            L * 4 * C,       // fcb
            L * C * (4 * C), // fcprojw
            L * C,           // fcprojb
            C,               // lnfw
            C,               // lnfb
        ];

        // count the number of paramaters
        let num_parameters = param_sizes.iter().sum::<usize>();
        let mut params_memory = vec![0.0; num_parameters];
        // read from file
        model_file
            .read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    params_memory.as_mut_ptr() as *mut u8,
                    num_parameters * std::mem::size_of::<f32>(),
                )
            })
            .expect("Error reading model file");

        let params = ParameterTensors::new(params_memory, &param_sizes);

        GPT2 {
            config,
            params,
            acts: ActivationTensors::default(),
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            num_activations: 0,
            batch_size: 0,
            seq_len: 0,
        }
    }

    fn forward(&mut self, inputs: &[u32], B: usize, T: usize) {
        // convenience parameters
        let V = self.config.vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // record the current B,T as well
        self.batch_size = B;
        self.seq_len = T;
        // and now allocate the space
        self.act_sizes[0] = B * T * C; // encoded
        self.act_sizes[1] = L * B * T * C; // ln1
        self.act_sizes[2] = L * B * T; // ln1_mean
        self.act_sizes[3] = L * B * T; // ln1_rstd
        self.act_sizes[4] = L * B * T * 3 * C; // qkv
        self.act_sizes[5] = L * B * T * C; // atty
        self.act_sizes[6] = L * B * NH * T * T; // preatt
        self.act_sizes[7] = L * B * NH * T * T; // att
        self.act_sizes[8] = L * B * T * C; // attproj
        self.act_sizes[9] = L * B * T * C; // residual2
        self.act_sizes[10] = L * B * T * C; // ln2
        self.act_sizes[11] = L * B * T; // ln2_mean
        self.act_sizes[12] = L * B * T; // ln2_rstd
        self.act_sizes[13] = L * B * T * 4 * C; // fch
        self.act_sizes[14] = L * B * T * 4 * C; // fch_gelu
        self.act_sizes[15] = L * B * T * C; // fcproj
        self.act_sizes[16] = L * B * T * C; // residual3
        self.act_sizes[17] = B * T * C; // lnf
        self.act_sizes[18] = B * T; // lnf_mean
        self.act_sizes[19] = B * T; // lnf_rstd
        self.act_sizes[20] = B * T * V; // logits
        self.act_sizes[21] = B * T * V; // probs
        self.act_sizes[22] = B * T; // losses

        let mut num_activations = 0;
        for i in 0..NUM_ACTIVATION_TENSORS {
            num_activations += self.act_sizes[i];
        }
        self.num_activations = num_activations;

        self.acts = ActivationTensors::new(&self.act_sizes);

        let params = &self.params;
        let acts = &mut self.acts;

        encoder_forward(&mut acts.encoded, inputs, &params.wte, &params.wpe, B, T, C);

        for l in 0..L {
            let residual = if l == 0 {
                &acts.encoded
            } else {
                &acts.residual3[(l - 1) * B * T * C..l * B * T * C].to_vec()
            };
            // get the pointers of the weights for this layer
            let l_ln1w = &params.ln1w[l * C..];
            let l_ln1b = &params.ln1b[l * C..];
            let l_qkvw = &params.qkvw[l * 3 * C * C..];
            let l_qkvb = &params.qkvb[l * 3 * C..];
            let l_attprojw = &params.attprojw[l * C * C..];
            let l_attprojb = &params.attprojb[l * C..];
            let l_ln2w = &params.ln2w[l * C..];
            let l_ln2b = &params.ln2b[l * C..];
            let l_fcw = &params.fcw[l * 4 * C * C..];
            let l_fcb = &params.fcb[l * 4 * C..];
            let l_fcprojw = &params.fcprojw[l * C * 4 * C..];
            let l_fcprojb = &params.fcprojb[l * C..];

            // get the pointers of the activations for this layer
            let l_ln1 = &mut acts.ln1[l * B * T * C..];
            let l_ln1_mean = &mut acts.ln1_mean[l * B * T..];
            let l_ln1_rstd = &mut acts.ln1_rstd[l * B * T..];
            let l_qkv = &mut acts.qkv[l * B * T * 3 * C..];
            let l_atty = &mut acts.atty[l * B * T * C..];
            let l_preatt = &mut acts.preatt[l * B * NH * T * T..];
            let l_att = &mut acts.att[l * B * NH * T * T..];
            let l_attproj = &mut acts.attproj[l * B * T * C..];
            let l_residual2 = &mut acts.residual2[l * B * T * C..];
            let l_ln2 = &mut acts.ln2[l * B * T * C..];
            let l_ln2_mean = &mut acts.ln2_mean[l * B * T..];
            let l_ln2_rstd = &mut acts.ln2_rstd[l * B * T..];
            let l_fch = &mut acts.fch[l * B * T * 4 * C..];
            let l_fch_gelu = &mut acts.fch_gelu[l * B * T * 4 * C..];
            let l_fcproj = &mut acts.fcproj[l * B * T * C..];
            let l_residual3 = &mut acts.residual3[l * B * T * C..];

            // now do the forward pass
            layernorm_forward(
                l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C,
            );
            matmul_forward(l_qkv, l_ln1, l_qkvw, Some(l_qkvb), B, T, C, 3 * C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, Some(l_attprojb), B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B * T * C);
            layernorm_forward(
                l_ln2,
                l_ln2_mean,
                l_ln2_rstd,
                l_residual2,
                l_ln2w,
                l_ln2b,
                B,
                T,
                C,
            );
            matmul_forward(l_fch, l_ln2, l_fcw, Some(l_fcb), B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
            matmul_forward(
                l_fcproj,
                l_fch_gelu,
                l_fcprojw,
                Some(l_fcprojb),
                B,
                T,
                4 * C,
                C,
            );
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }
        let residual = &acts.residual3[(L - 1) * B * T * C..];
        layernorm_forward(
            &mut acts.lnf,
            &mut acts.lnf_mean,
            &mut acts.lnf_rstd,
            residual,
            &params.lnfw,
            &params.lnfb,
            B,
            T,
            C,
        );
        matmul_forward(&mut acts.logits, &acts.lnf, &params.wte, None, B, T, C, V);
        softmax_forward(&mut acts.probs, &acts.logits, B, T, V);
    }
}

struct GPT2Config {
    /// vocab size, e.g. 50257
    vocab_size: usize,
    /// number of layers, e.g. 12
    num_layers: usize,
    /// number of heads in attention, e.g. 12
    num_heads: usize,
    /// number of channels, e.g. 768
    channels: usize,
}

struct ParameterTensors {
    /// (V, C)
    wte: Vec<f32>,
    /// (maxT, C)
    wpe: Vec<f32>,
    /// (L, C)
    ln1w: Vec<f32>,
    /// (L, C)
    ln1b: Vec<f32>,
    /// (L, 3*C, C)
    qkvw: Vec<f32>,
    /// (L, 3*C)
    qkvb: Vec<f32>,
    /// (L, C, C)
    attprojw: Vec<f32>,
    /// (L, C)
    attprojb: Vec<f32>,
    /// (L, C)
    ln2w: Vec<f32>,
    /// (L, C)
    ln2b: Vec<f32>,
    /// (L, 4*C, C)
    fcw: Vec<f32>,
    /// (L, 4*C)
    fcb: Vec<f32>,
    /// (L, C, 4*C)
    fcprojw: Vec<f32>,
    /// (L, C)
    fcprojb: Vec<f32>,
    /// (C)
    lnfw: Vec<f32>,
    /// (C)
    lnfb: Vec<f32>,
}

impl ParameterTensors {
    fn new(params: Vec<f32>, param_sizes: &[usize; NUM_PARAMETER_TENSORS]) -> Self {
        let mut offset = 0;
        let mut param_tensors = ParameterTensors {
            wte: Vec::new(),
            wpe: Vec::new(),
            ln1w: Vec::new(),
            ln1b: Vec::new(),
            qkvw: Vec::new(),
            qkvb: Vec::new(),
            attprojw: Vec::new(),
            attprojb: Vec::new(),
            ln2w: Vec::new(),
            ln2b: Vec::new(),
            fcw: Vec::new(),
            fcb: Vec::new(),
            fcprojw: Vec::new(),
            fcprojb: Vec::new(),
            lnfw: Vec::new(),
            lnfb: Vec::new(),
        };
        for (i, size) in param_sizes.iter().enumerate() {
            let slice = &params[offset..offset + size];
            offset += size;
            match i {
                0 => param_tensors.wte = slice.to_vec(),
                1 => param_tensors.wpe = slice.to_vec(),
                2 => param_tensors.ln1w = slice.to_vec(),
                3 => param_tensors.ln1b = slice.to_vec(),
                4 => param_tensors.qkvw = slice.to_vec(),
                5 => param_tensors.qkvb = slice.to_vec(),
                6 => param_tensors.attprojw = slice.to_vec(),
                7 => param_tensors.attprojb = slice.to_vec(),
                8 => param_tensors.ln2w = slice.to_vec(),
                9 => param_tensors.ln2b = slice.to_vec(),
                10 => param_tensors.fcw = slice.to_vec(),
                11 => param_tensors.fcb = slice.to_vec(),
                12 => param_tensors.fcprojw = slice.to_vec(),
                13 => param_tensors.fcprojb = slice.to_vec(),
                14 => param_tensors.lnfw = slice.to_vec(),
                15 => param_tensors.lnfb = slice.to_vec(),
                _ => panic!("Bad parameter tensor index"),
            }
        }
        param_tensors
    }
}

#[derive(Default)]
struct ActivationTensors {
    /// (B, T, C)
    encoded: Vec<f32>,
    /// (L, B, T, C)
    ln1: Vec<f32>,
    /// (L, B, T)
    ln1_mean: Vec<f32>,
    /// (L, B, T)
    ln1_rstd: Vec<f32>,
    /// (L, B, T, 3*C)
    qkv: Vec<f32>,
    /// (L, B, T, C)
    atty: Vec<f32>,
    /// (L, B, NH, T, T)
    preatt: Vec<f32>,
    /// (L, B, NH, T, T)
    att: Vec<f32>,
    /// (L, B, T, C)
    attproj: Vec<f32>,
    /// (L, B, T, C)
    residual2: Vec<f32>,
    /// (L, B, T, C)
    ln2: Vec<f32>,
    /// (L, B, T)
    ln2_mean: Vec<f32>,
    /// (L, B, T)
    ln2_rstd: Vec<f32>,
    /// (L, B, T, 4*C)
    fch: Vec<f32>,
    /// (L, B, T, 4*C)
    fch_gelu: Vec<f32>,
    /// (L, B, T, C)
    fcproj: Vec<f32>,
    /// (L, B, T, C)
    residual3: Vec<f32>,
    /// (B, T, C)
    lnf: Vec<f32>,
    /// (B, T)
    lnf_mean: Vec<f32>,
    /// (B, T)
    lnf_rstd: Vec<f32>,
    /// (B, T, V)
    logits: Vec<f32>,
    /// (B, T, V)
    probs: Vec<f32>,
}

impl ActivationTensors {
    fn new(act_sizes: &[usize; NUM_ACTIVATION_TENSORS]) -> Self {
        ActivationTensors {
            encoded: vec![0.0; act_sizes[0]],
            ln1: vec![0.0; act_sizes[1]],
            ln1_mean: vec![0.0; act_sizes[2]],
            ln1_rstd: vec![0.0; act_sizes[3]],
            qkv: vec![0.0; act_sizes[4]],
            atty: vec![0.0; act_sizes[5]],
            preatt: vec![0.0; act_sizes[6]],
            att: vec![0.0; act_sizes[7]],
            attproj: vec![0.0; act_sizes[8]],
            residual2: vec![0.0; act_sizes[9]],
            ln2: vec![0.0; act_sizes[10]],
            ln2_mean: vec![0.0; act_sizes[11]],
            ln2_rstd: vec![0.0; act_sizes[12]],
            fch: vec![0.0; act_sizes[13]],
            fch_gelu: vec![0.0; act_sizes[14]],
            fcproj: vec![0.0; act_sizes[15]],
            residual3: vec![0.0; act_sizes[16]],
            lnf: vec![0.0; act_sizes[17]],
            lnf_mean: vec![0.0; act_sizes[18]],
            lnf_rstd: vec![0.0; act_sizes[19]],
            logits: vec![0.0; act_sizes[20]],
            probs: vec![0.0; act_sizes[21]],
        }
    }
}

fn encoder_forward(
    out: &mut [f32],
    inp: &[u32],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let out_bt = &mut out[b * T * C + t * C..];
            let ix = inp[b * T + t] as usize;
            let wte_ix = &wte[ix * C..];
            let wpe_t = &wpe[t * C..];
            for i in 0..C {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    let eps = 1e-5;
    for b in 0..B {
        for t in 0..T {
            // seed to the input position inp[b, t, :]
            let x = &inp[b * T * C + t * C..];
            // calculate the mean
            let m = x.iter().take(C).sum::<f32>() / C as f32;
            // calculate the variance
            let v = x.iter().take(C).map(|v| (v - m) * (v - m)).sum::<f32>() / C as f32;
            // calculate the rstd
            let s = 1.0 / (v + eps).sqrt();
            // calculate the output
            let out_bt = &mut out[b * T * C + t * C..];
            for i in 0..C {
                let n = s * (x[i] - m); // normalized
                let o = n * weight[i] + bias[i]; // scaled and biased
                out_bt[i] = o; // store the output
            }
            // cache the mean and rstd for backprop
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], n: usize) {
    for i in 0..n {
        out[i] = inp1[i] + inp2[i];
    }
}

/// output: probs are (B, T, V) of the probabilities (sums to 1.0 for each B, T)
/// input: logits are (B, T, V) of the unnormalized probabilities
fn softmax_forward(probs: &mut [f32], logits: &[f32], B: usize, T: usize, V: usize) {
    for b in 0..B {
        for t in 0..T {
            let logits_bt = &logits[b * T * V + t * V..];
            let probs_bt = &mut probs[b * T * V + t * V..];

            // maxval is only calculated to avoid numerical instability
            let maxval = logits_bt
                .iter()
                .take(V)
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut sum = 0.0;
            for i in 0..V {
                probs_bt[i] = (logits_bt[i] - maxval).exp();
                sum += probs_bt[i];
            }

            probs_bt.iter_mut().take(V).for_each(|p| *p /= sum);
        }
    }
}

fn gelu_forward(out: &mut [f32], inp: &[f32], n: usize) {
    let pi = std::f32::consts::PI;
    let GELU_SCALING_FACTOR = (2.0 / pi).sqrt();
    for i in 0..n {
        let x = inp[i];
        let cube = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + (GELU_SCALING_FACTOR * (x + cube)).tanh());
    }
}

fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    // output channels
    OC: usize,
) {
    let bias = match bias {
        Some(bias) => bias,
        None => &vec![0.0; OC],
    };
    #[cfg(not(feature = "noparallel"))]
    {
        let out_bt = (0..B)
            .into_par_iter()
            .flat_map(|b| {
                (0..T).into_par_iter().flat_map(move |t| {
                    let inp_bt = &inp[b * T * C + t * C..];
                    (0..OC).into_par_iter().map(move |o| {
                        let wrow = &weight[o * C..(o + 1) * C];
                        wrow.iter()
                            .zip(inp_bt.iter())
                            .map(|(&w, &i)| w * i)
                            .sum::<f32>()
                            + bias[o]
                    })
                })
            })
            .collect::<Vec<f32>>();
        out[..T * B * OC].copy_from_slice(&out_bt);
    }

    #[cfg(feature = "noparallel")]
    {
        for b in 0..B {
            for t in 0..T {
                let out_bt = &mut out[b * T * OC + t * OC..];
                let inp_bt = &inp[b * T * C + t * C..];
                for o in 0..OC {
                    let mut val = bias[o];
                    let wrow = &weight[o * C..];
                    for i in 0..C {
                        val += wrow[i] * inp_bt[i];
                    }
                    out_bt[o] = val;
                }
            }
        }
    }
}

fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = 3 * C;
    let hs = C / NH;
    let scale = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let query_t = &inp[b * T * C3 + t * C3 + h * hs..];
                let preatt_bth = &mut preatt[b * NH * T * T + h * T * T + t * T..];
                let att_bth = &mut att[b * NH * T * T + h * T * T + t * T..];

                // pass 1: calculate the query dot key and maxval
                let mut maxval = f32::NEG_INFINITY;
                for t2 in 0..=t {
                    let key_t2 = &inp[(b * T * C3 + t2 * C3 + h * hs + C)..];

                    // query dot key
                    let mut val = 0.0;
                    for i in 0..hs {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    maxval = maxval.max(val);

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                let mut expsum = 0.0;
                for t2 in 0..=t {
                    let expv = (preatt_bth[t2] - maxval).exp();
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // pass 3: normalize to get the softmax
                for (t2, item) in att_bth.iter_mut().enumerate().take(T) {
                    if t2 <= t {
                        *item *= expsum_inv;
                    } else {
                        *item = 0.0;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                let out_bth = &mut out[b * T * C + t * C + h * hs..];
                for i in out_bth.iter_mut().take(hs) {
                    *i = 0.0;
                }
                for t2 in 0..=t {
                    let valut_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + 2 * C..];
                    let att_btht2 = att_bth[t2];
                    for i in 0..hs {
                        out_bth[i] += att_btht2 * valut_t2[i];
                    }
                }
            }
        }
    }
}

fn sample_mult(probabilities: &[f32], n: usize) -> usize {
    let mut cdf = 0.0;
    let coin = 0.5;
    for (i, item) in probabilities.iter().enumerate().take(n) {
        cdf += *item;
        if cdf >= coin {
            return i;
        }
    }
    n - 1
}

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().expect("No command provided");

    let bpe = r50k_base().unwrap();
    let prefix = bpe.encode_ordinary(&command);

    let mut model = GPT2::build_from_checkpoint("./gpt2_124M.bin");
    let mut tokens: Vec<u32> = vec![0; N];

    for i in 0..N {
        if i < prefix.len() {
            tokens[i] = prefix[i];
        } else {
            tokens[i] = GPT2_EOT;
        }
    }

    print!("{}", command);
    for t in prefix.len()..N {
        model.forward(&tokens, 1, t);
        let probs = &model.acts.probs[(t - 1) * model.config.vocab_size..];
        let next_token = sample_mult(probs, model.config.vocab_size);
        tokens[t] = next_token as u32;
        print!("{}", bpe.decode(vec![tokens[t]]).unwrap());
        std::io::stdout().flush().unwrap();
    }
    println!();
}
