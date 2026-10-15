use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
use std::collections::HashSet;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const NAMES_URL: &str = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";

type ValueRef = Rc<RefCell<Value>>;
type Matrix = Vec<Vec<ValueRef>>;

/// Let there be Autograd to recursively apply the chain rule through a computation graph
#[derive(Debug, Clone)]
struct Value {
    /// scalar value of this node calculated during forward pass
    data: f64,
    /// derivative of the loss w.r.t. this node, calculated in backward pass
    grad: f64,
    /// children of this node in the computation graph
    children: Vec<ValueRef>,
    /// local derivative of this node w.r.t. its children
    local_grads: Vec<f64>,
}

impl Value {
    fn new(data: f64) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        }))
    }
}

trait ValueOps {
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn pow(&self, other: f64) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn truediv(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn log(&self) -> Self;
    fn exp(&self) -> Self;
    fn relu(&self) -> Self;
    fn backward(&self);
}

impl ValueOps for ValueRef {
    fn add(&self, other: &Self) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data + other.borrow().data,
            grad: 0.0,
            children: vec![self.clone(), other.clone()],
            local_grads: vec![1.0, 1.0],
        }))
    }

    fn mul(&self, other: &Self) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data * other.borrow().data,
            grad: 0.0,
            children: vec![self.clone(), other.clone()],
            local_grads: vec![other.borrow().data, self.borrow().data],
        }))
    }

    fn pow(&self, other: f64) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data.powf(other),
            grad: 0.0,
            children: vec![self.clone()],
            local_grads: vec![other * self.borrow().data.powf(other - 1.0)],
        }))
    }

    fn neg(&self) -> Self {
        self.mul(&Value::new(-1.0))
    }

    fn sub(&self, other: &Self) -> Self {
        self.add(&Value::new(-1.0).mul(other))
    }

    fn truediv(&self, other: &Self) -> Self {
        self.mul(&other.pow(-1.0))
    }

    fn relu(&self) -> Self {
        Rc::new(RefCell::new(Value {
            data: f64::max(0.0, self.borrow().data),
            grad: 0.0,
            children: vec![self.clone()],
            local_grads: vec![f64::from(self.borrow().data > 0.0)],
        }))
    }

    fn log(&self) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data.ln(),
            grad: 0.0,
            children: vec![self.clone()],
            local_grads: vec![1.0 / self.borrow().data],
        }))
    }

    fn exp(&self) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data.exp(),
            grad: 0.0,
            children: vec![self.clone()],
            local_grads: vec![self.borrow().data.exp()],
        }))
    }

    fn backward(&self) {
        fn build_topo(
            v: &ValueRef,
            visited: &mut HashSet<*const RefCell<Value>>,
            topo: &mut Vec<ValueRef>,
        ) {
            let addr = Rc::as_ptr(v);
            if !visited.contains(&addr) {
                visited.insert(addr);
                for child in &v.borrow().children {
                    build_topo(child, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        let mut topo: Vec<ValueRef> = Vec::new();
        let mut visited: HashSet<*const RefCell<Value>> = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

        self.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            for (child, local_grad) in v.borrow().children.iter().zip(&v.borrow().local_grads) {
                child.borrow_mut().grad += local_grad * v.borrow().grad;
            }
        }
    }
}

const MEAN: f64 = 0.0;
const STD: f64 = 0.08;

fn matrix(rng: &mut StdRng, mean: f64, std: f64, nout: usize, nin: usize) -> Matrix {
    let normal = Normal::new(mean, std).unwrap();
    (0..nout)
        .map(|_| (0..nin).map(|_| Value::new(normal.sample(rng))).collect())
        .collect()
}

fn sum(x: impl IntoIterator<Item = ValueRef>) -> ValueRef {
    x.into_iter().fold(Value::new(0.0), |acc, x| acc.add(&x))
}

// Define the model architecture: a function mapping tokens and parameters to logits over what comes next
fn linear(x: &Vec<ValueRef>, w: &Matrix) -> Vec<ValueRef> {
    w.iter()
        .map(|wo| sum(wo.iter().zip(x).map(|(wi, xi)| wi.mul(xi))))
        .collect()
}

fn softmax(logits: &Vec<ValueRef>) -> Vec<ValueRef> {
    let max_val: ValueRef = Value::new(
        logits
            .iter()
            .map(|v| v.borrow().data)
            .max_by(|x, y| x.total_cmp(y))
            .unwrap(),
    );
    let exps: Vec<ValueRef> = logits.iter().map(|val| val.sub(&max_val).exp()).collect();
    let total: ValueRef = sum(exps.iter().cloned());
    exps.iter().map(|e| e.truediv(&total)).collect()
}

fn rmsnorm(x: &Vec<ValueRef>, eps: f64) -> Vec<ValueRef> {
    let ms = sum(x.iter().map(|xi| xi.mul(xi))).truediv(&Value::new(x.len() as f64));
    let scale = ms.add(&Value::new(eps)).pow(-0.5);
    x.iter().map(|xi| xi.mul(&scale)).collect()
}

fn gpt(
    token_id: usize,
    pos_id: usize,
    n_layer: usize,
    n_head: usize,
    head_dim: usize,
    keys: &mut Vec<Matrix>,
    values: &mut Vec<Matrix>,
    state_dict: &HashMap<String, Matrix>,
) -> Vec<ValueRef> {
    // token embedding
    let tok_emb: &Vec<ValueRef> = &state_dict["wte"][token_id];
    // position embedding
    let pos_emb: &Vec<ValueRef> = &state_dict["wpe"][pos_id];
    // joint token and position embedding
    let mut x: Vec<ValueRef> = tok_emb.iter().zip(pos_emb).map(|(t, p)| t.add(p)).collect();
    // note: not redundant due to backward pass via the residual connection
    x = rmsnorm(&x, 1e-5);

    for li in 0..n_layer {
        // 1) Multi-head Attention block
        let x_residual = x.clone();
        x = rmsnorm(&x, 1e-5);
        let q: Vec<ValueRef> = linear(&x, &state_dict[&format!("layer{li}.attn_wq")]);
        let k: Vec<ValueRef> = linear(&x, &state_dict[&format!("layer{li}.attn_wk")]);
        let v: Vec<ValueRef> = linear(&x, &state_dict[&format!("layer{li}.attn_wv")]);
        keys[li].push(k);
        values[li].push(v);

        let mut x_attn: Vec<ValueRef> = Vec::with_capacity(n_head * head_dim);
        for h in 0..n_head {
            let hs = h * head_dim;
            let q_h = &q[hs..hs + head_dim];
            let k_h: Vec<&[ValueRef]> = keys[li].iter().map(|ki| &ki[hs..hs + head_dim]).collect();
            let v_h: Vec<&[ValueRef]> =
                values[li].iter().map(|vi| &vi[hs..hs + head_dim]).collect();
            let attn_logits = (0..k_h.len())
                .map(|t| {
                    let dot_product: ValueRef = sum((0..head_dim).map(|j| q_h[j].mul(&k_h[t][j])));
                    dot_product.truediv(&Value::new((head_dim as f64).sqrt()))
                })
                .collect();
            let attn_weights = softmax(&attn_logits);
            let head_out: Vec<ValueRef> = (0..head_dim)
                .map(|j| sum((0..v_h.len()).map(|t| attn_weights[t].mul(&v_h[t][j]))))
                .collect();
            x_attn.extend(head_out);
        }
        x = linear(&x_attn, &state_dict[&format!("layer{li}.attn_wo")]);
        x = x.iter().zip(x_residual).map(|(a, b)| a.add(&b)).collect();

        // 2) MLP block
        let x_residual = x.clone();
        x = rmsnorm(&x, 1e-5);
        x = linear(&x, &state_dict[&format!("layer{li}.mlp_fc1")]);
        x = x.iter().map(ValueRef::relu).collect();
        x = linear(&x, &state_dict[&format!("layer{li}.mlp_fc2")]);
        x = x.iter().zip(x_residual).map(|(a, b)| a.add(&b)).collect();
    }

    let logits = linear(&x, &state_dict["lm_head"]);
    logits
}

pub fn train(names: String) {
    // Let there be order among chaos
    let mut rng = StdRng::seed_from_u64(42);
    // Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
    let mut docs: Vec<&str> = names
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect();
    docs.shuffle(&mut rng);
    log(&format!("num docs: {}", docs.len()));

    // Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    // unique characters in the dataset become token ids 0..n-1
    let uchars: BTreeSet<char> = BTreeSet::from_iter(docs.iter().flat_map(|s| s.chars()));
    let uchars: Vec<&char> = uchars.iter().collect();

    // token id for a special Beginning of Sequence (BOS) token
    #[allow(non_snake_case)]
    let BOS = uchars.len();

    // total number of unique tokens, +1 is for BOS
    let vocab_size = uchars.len() + 1;
    log(&format!("vocab size: {}", vocab_size));

    // Initialize the parameters, to store the knowledge of the model

    // depth of the transformer neural network (number of layers)
    let n_layer = 1;
    // width of the network (embedding dimension)
    let n_embd = 16;
    // maximum context length of the attention window (note: the longest name is 15 characters)
    let block_size = 16;
    // number of attention heads
    let n_head = 4;
    // derived dimension of each head
    let head_dim = n_embd / n_head;

    let mut state_dict: HashMap<String, Matrix> = HashMap::from([
        (
            "wte".into(),
            matrix(&mut rng, MEAN, STD, vocab_size, n_embd),
        ),
        (
            "wpe".into(),
            matrix(&mut rng, MEAN, STD, block_size, n_embd),
        ),
        (
            "lm_head".into(),
            matrix(&mut rng, MEAN, STD, vocab_size, n_embd),
        ),
    ]);

    for i in 0..n_layer {
        state_dict.extend([
            (
                format!("layer{i}.attn_wq"),
                matrix(&mut rng, MEAN, STD, n_embd, n_embd),
            ),
            (
                format!("layer{i}.attn_wk"),
                matrix(&mut rng, MEAN, STD, n_embd, n_embd),
            ),
            (
                format!("layer{i}.attn_wv"),
                matrix(&mut rng, MEAN, STD, n_embd, n_embd),
            ),
            (
                format!("layer{i}.attn_wo"),
                matrix(&mut rng, MEAN, STD, n_embd, n_embd),
            ),
            (
                format!("layer{i}.mlp_fc1"),
                matrix(&mut rng, MEAN, STD, 4 * n_embd, n_embd),
            ),
            (
                format!("layer{i}.mlp_fc2"),
                matrix(&mut rng, MEAN, STD, n_embd, 4 * n_embd),
            ),
        ]);
    }

    let params: Vec<&ValueRef> = state_dict.values().flatten().flatten().collect();
    log(&format!("num params: {}", params.len()));

    // Let there be Adam, the blessed optimizer and its buffers
    let (learning_rate, beta1, beta2, eps_adam) = (0.01, 0.85, 0.99, 1e-8);
    let mut m = vec![0.0; params.len()]; // first moment buffer
    let mut v = vec![0.0; params.len()]; // second moment buffer

    // Repeat in sequence
    let num_steps = 1000;
    for step in 0..num_steps {
        // Take single document, tokenize it, surround it with BOS special token on both sides
        let doc: &str = docs[step % docs.len()];
        let mut tokens = vec![BOS];
        tokens.extend(
            doc.chars()
                .map(|ch| uchars.iter().position(|&&c| c == ch).unwrap()),
        );
        tokens.push(BOS);
        let n = usize::min(block_size, tokens.len() - 1);

        // Forward the token sequence through the model, building up the computation graph all the way to the loss
        let (mut keys, mut values): (Vec<Matrix>, Vec<Matrix>) =
            (vec![Vec::new(); n_layer], vec![Vec::new(); n_layer]);
        let mut losses: Vec<ValueRef> = Vec::with_capacity(n);
        for pos_id in 0..n {
            let (token_id, target_id) = (tokens[pos_id], tokens[pos_id + 1]);
            let logits = gpt(
                token_id,
                pos_id,
                n_layer,
                n_head,
                head_dim,
                &mut keys,
                &mut values,
                &state_dict,
            );
            let probs = softmax(&logits);
            let loss_t: ValueRef = probs[target_id].log().neg();
            losses.push(loss_t);
        }

        // final average loss over the document sequence. May yours be low.
        let loss: ValueRef = Value::new(1.0 / n as f64).mul(&sum(losses));

        // Backward the loss, calculating the gradients with respect to all model parameters
        loss.backward();

        // Adam optimizer update: update the model parameters based on the corresponding gradients
        // linear learning rate decay
        let lr_t = learning_rate * (1.0 - (step as f64) / (num_steps as f64));
        for (i, p) in params.iter().enumerate() {
            m[i] = beta1 * m[i] + (1.0 - beta1) * p.borrow().grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * p.borrow().grad.powi(2);
            let m_hat = m[i] / (1.0 - beta1.powi(step as i32 + 1));
            let v_hat = v[i] / (1.0 - beta2.powi(step as i32 + 1));
            p.borrow_mut().data -= lr_t * m_hat / (v_hat.powf(0.5) + eps_adam);
            p.borrow_mut().grad = 0.0;
        }

        if (step + 1) % 50 == 0 || step == 0 {
            log(&format!(
                "step {:4} / {:4} | loss {:.4}",
                step + 1,
                num_steps,
                loss.borrow().data
            ));
        }
    }

    // Inference: may the model babble back to us

    // in (0, 1], control the "creativity" of generated text, low to high
    let temperature = 0.5;
    log("--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let (mut keys, mut values) = (vec![Vec::new(); n_layer], vec![Vec::new(); n_layer]);
        let mut token_id = BOS;
        let mut sample = Vec::new();
        for pos_id in 0..block_size {
            let logits = gpt(
                token_id,
                pos_id,
                n_layer,
                n_head,
                head_dim,
                &mut keys,
                &mut values,
                &state_dict,
            );
            let probs: Vec<ValueRef> = softmax(
                &logits
                    .iter()
                    .map(|l| l.truediv(&Value::new(temperature)))
                    .collect(),
            );
            token_id = *(0..vocab_size)
                .collect::<Vec<usize>>()
                .choose_weighted(&mut rng, |&i| probs[i].borrow().data)
                .unwrap();
            if token_id == BOS {
                break;
            }
            sample.push(uchars[token_id]);
        }
        let name: String = sample.into_iter().collect();
        log(&format!("sample {:2}: {}", sample_idx + 1, name));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn log(msg: &str) {
    println!("{msg}");
}
#[cfg(target_arch = "wasm32")]
fn log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}

#[cfg(not(target_arch = "wasm32"))]
pub fn fetch_names() -> Result<String, Box<dyn std::error::Error>> {
    if std::path::Path::new("input.txt").exists() {
        Ok(std::fs::read_to_string("input.txt")?)
    } else {
        let text = reqwest::blocking::get(NAMES_URL)?.text()?;
        std::fs::write("input.txt", &text)?;
        Ok(text)
    }
}
#[cfg(target_arch = "wasm32")]
pub async fn fetch_names() -> Result<String, JsValue> {
    let text = web_sys::window()
        .unwrap()
        .fetch_with_str(NAMES_URL)
        .await?
        .dyn_into::<web_sys::Response>()?
        .text()?
        .await?
        .as_string()
        .unwrap();
    Ok(text)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
/// Entrypoint function for WASM running in the browser
pub async fn run() -> Result<(), JsValue> {
    train(fetch_names().await?);
    Ok(())
}
