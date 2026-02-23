use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::iter::Sum;
use std::path::Path;
use std::rc::Rc;
use std::{collections::HashSet, ops::Add, ops::Mul, vec};

// NOTE: I define other type aliases below too; should they all be grouped at the top?
// Reference is https://github.com/nrc/r4cppp/blob/master/graphs/README.md for the trick of using an Rc<RefCell<Value>>
type ValueRef = Rc<RefCell<Value>>;

/// TODO: Consider dtypes: can use f16/f32 (or make generic)
/// Needs some thinking, but should be able to couple the children and local_grads together
/// - using two parallel vecs to store associated data is just gross.
/// Since there's only ever max two children of a given node, we could probably also use an Enum
/// and not use a Vec. But then we'd have to pattern match everywhere and the code would be more verbose
/// If the graph is acyclic (I think it's not due to residual maybe?), we could even get away with a Box instead of a Rc maybe? (Not 100% sure on that...)
#[derive(Debug, Clone)]
struct Value {
    data: f64,               // scalar value of this node calculated during forward pass
    grad: f64,               // derivative of the loss w.r.t. this node, calculated in backward pass
    children: Vec<ValueRef>, // children of this node in the computation graph
    local_grads: Vec<f64>,   // local derivative of this node w.r.t. its children
}

impl Value {
    /// I can't impl ValueRef directly, but should this return Rc<Value> directly?
    /// Or a reference &Rc<ValueRef> instead?
    fn new(data: f64) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        }))
    }

    /// I don't love that I can't directly make these &self methods
    /// I can wrap ValueRef in a new struct, but might not be clean either
    /// There's also some redundancy in these methods that can be factored out
    /// Ideally add and sub should be an impl for operator overload, to make syntax more concise
    fn add(self_: &ValueRef, other: &ValueRef) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: self_.borrow().data + other.borrow().data,
            grad: 0.0,
            children: vec![self_.clone(), other.clone()],
            local_grads: vec![1.0, 1.0],
        }))
    }

    fn mul(self_: &ValueRef, other: &ValueRef) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: self_.borrow().data * other.borrow().data,
            grad: 0.0,
            children: vec![self_.clone(), other.clone()],
            local_grads: vec![other.borrow().data, self_.borrow().data],
        }))
    }

    fn pow(self_: &ValueRef, other: f64) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: self_.borrow().data.powf(other),
            grad: 0.0,
            children: vec![self_.clone()],
            local_grads: vec![other * self_.borrow().data.powf(other - 1.0)],
        }))
    }

    fn log(self_: &ValueRef) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: self_.borrow().data.ln(),
            grad: 0.0,
            children: vec![self_.clone()],
            local_grads: vec![1.0 / self_.borrow().data],
        }))
    }

    fn exp(self_: &ValueRef) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: self_.borrow().data.exp(),
            grad: 0.0,
            children: vec![self_.clone()],
            local_grads: vec![self_.borrow().data.exp()],
        }))
    }

    fn relu(self_: &ValueRef) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: f64::max(0.0, self_.borrow().data),
            grad: 0.0,
            children: vec![self_.clone()],
            local_grads: vec![f64::from(self_.borrow().data > 0.0)], // TODO: cleaner expression?
        }))
    }

    fn neg(self_: &ValueRef) -> ValueRef {
        Self::mul(self_, &Value::new(-1.0))
    }

    // NOTE: python implementation defines the right-associative arithmetic operations, but we elide them
    // fn __radd__
    // fn __rsub__
    // fn __rmul__
    // fn __rtruediv__

    fn sub(self_: &ValueRef, other: &ValueRef) -> ValueRef {
        Self::add(self_, &Self::mul(&Value::new(-1.0), other))
    }

    fn truediv(self_: &ValueRef, other: &ValueRef) -> ValueRef {
        Self::mul(self_, &Self::pow(other, -1.0))
    }

    /// This is a /bit/ gnarly: we can't derive Hash on the Value struct, since the f64
    /// doesn't compare Eq, because of NaNs (not-reflective)
    /// Thus, we can't store the Value nodes in a HashSet directly
    /// But, we can store pointer to a RefCell<Value>; effectively, we're using the pointer address
    /// as a unique object ID, as opposed to a Hash, or, using an actual globally unique ID
    /// Still feels a bit messy though, but idk, maybe is idiomatic
    fn backward(self_: &ValueRef) {
        let mut topo: Vec<ValueRef> = Vec::new();
        let mut visited: HashSet<*const RefCell<Value>> = HashSet::new();
        Self::build_topo(self_, &mut visited, &mut topo);

        self_.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            for (child, local_grad) in v.borrow().children.iter().zip(&v.borrow().local_grads) {
                child.borrow_mut().grad += local_grad * v.borrow().grad;
            }
        }
    }

    fn build_topo(
        v: &ValueRef,
        visited: &mut HashSet<*const RefCell<Value>>,
        topo: &mut Vec<Rc<RefCell<Value>>>,
    ) {
        let addr = Rc::as_ptr(v);
        if !visited.contains(&addr) {
            visited.insert(addr);
            for child in &v.borrow().children {
                Self::build_topo(child, visited, topo);
            }
            topo.push(v.clone());
        }
    }
}

// Could also define a type Embedding = Vec<ValueRef>; (but not every Vec<ValueRef> is an embedding)
type Matrix = Vec<Vec<ValueRef>>; // I'd prefer to use a flat Vec with row-major ordering, but as Karpathy does, so be it
fn matrix(nout: usize, nin: usize) -> Matrix {
    let mut rng = rand::rng(); // Not really important, but how does one set seeds in Rust?
    let normal = Normal::new(0.0, 0.08).unwrap(); // ugh, magic numbers

    // I struggle with this, I do feel like nested for loops might be more readable
    // but a transliteration of the list comprehensions brings us here...
    (0..nout)
        .map(|_| {
            (0..nin)
                .map(|_| Value::new(normal.sample(&mut rng)))
                .collect()
        })
        .collect()
}

// Would be nicer to impl Sum directly, I feel
// impl Sum for Vec<ValueRef> {
//     fn sum<I: Iterator<Item = Self>>(std::iter: I) -> Self {
//         todo!()
//     }
// }
fn sum(x: &Vec<ValueRef>) -> ValueRef {
    x.iter()
        .fold(Value::new(0.0), |acc, x| Value::add(&acc, &x))
}

// Would need refactoring, but there's an awful lot of collects throughout
// Can I return raw iterators, and only collapse into a collect at the end?
// Would that needlessly complicate the type signatures to something dynamic'y though?
fn linear(x: &Vec<ValueRef>, w: &Matrix) -> Vec<ValueRef> {
    w.iter()
        .map(|wo| {
            sum(&wo
                .iter()
                .zip(x) // do I need x.iter?
                .map(|(wi, xi)| Value::mul(wi, xi))
                .collect())
        })
        .collect()
}

fn softmax(logits: &Vec<ValueRef>) -> Vec<ValueRef> {
    // assert!(logits.iter().all(|v| v.borrow().data.is_finite()));
    // We do some hackery to compute the max of a Vec<f64>
    let max_val: f64 = logits
        .iter()
        .map(|v| v.borrow().data)
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max_val: ValueRef = Value::new(max_val); // really need a nicer way than explictly casting
    let exps: Vec<ValueRef> = logits
        .iter()
        .map(|val| Value::exp(&Value::sub(val, &max_val)))
        .collect();
    let total: ValueRef = sum(&exps);

    exps.iter().map(|e| Value::truediv(&e, &total)).collect()
}

fn rmsnorm(x: &Vec<ValueRef>) -> Vec<ValueRef> {
    let ms = Value::truediv(
        &sum(&x.iter().map(|xi| Value::mul(xi, xi)).collect()),
        &Value::new(x.len() as f64),
    );
    let scale = Value::pow(&Value::add(&ms, &Value::new(1e-5)), -0.5); // more magic values...
    x.iter().map(|xi| Value::mul(xi, &scale)).collect()
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
    let tok_emb: &Vec<ValueRef> = &state_dict.get(&"wte".to_string()).unwrap()[token_id]; // into vs to_string vs to_owned vs ...
    let pos_emb: &Vec<ValueRef> = &state_dict.get(&"wpe".to_string()).unwrap()[pos_id];
    let mut x: Vec<ValueRef> = tok_emb
        .iter()
        .zip(pos_emb)
        .map(|(t, p)| Value::add(t, p))
        .collect();
    x = rmsnorm(&x);

    for li in 0..n_layer {
        // 1) Multi-head Attention block
        let x_residual = x.clone(); // Do I need to copy/clone? (I think the python is implictly doing that...)
        x = rmsnorm(&x);
        //NOTE for article; I got quite stuck here ^^
        let q: Vec<ValueRef> = linear(&x, state_dict.get(&format!("layer{li}.attn_wq")).unwrap());
        let k: Vec<ValueRef> = linear(&x, state_dict.get(&format!("layer{li}.attn_wk")).unwrap());
        let v: Vec<ValueRef> = linear(&x, state_dict.get(&format!("layer{li}.attn_wv")).unwrap());
        keys[li].push(k);
        values[li].push(v);

        let mut x_attn: Vec<ValueRef> = Vec::new();
        for h in 0..n_head {
            let hs = h * head_dim;
            let q_h = &q[hs..hs + head_dim];
            let k_h: Vec<&[ValueRef]> = keys[li].iter().map(|ki| &ki[hs..hs + head_dim]).collect();
            let v_h: Vec<&[ValueRef]> =
                values[li].iter().map(|vi| &vi[hs..hs + head_dim]).collect();
            let attn_logits: Vec<ValueRef> = (0..k_h.len())
                .map(|t| {
                    let dot_product: ValueRef = sum(&(0..head_dim)
                        .map(|j| Value::mul(&q_h[j], &k_h[t][j]))
                        .collect());

                    Value::truediv(&dot_product, &Value::new((head_dim as f64).sqrt()))
                })
                .collect();
            let attn_weights = softmax(&attn_logits);
            let head_out: Vec<ValueRef> = (0..head_dim)
                .map(|j| {
                    sum(&(0..v_h.len())
                        .map(|t| Value::mul(&attn_weights[t], &v_h[t][j]))
                        .collect())
                })
                .collect();
            x_attn.extend(head_out);
        }
        x = linear(
            &x_attn,
            &state_dict.get(&format!("layer{li}.attn_wo")).unwrap(),
        );
        x = x
            .iter()
            .zip(x_residual)
            .map(|(a, b)| Value::add(&a, &b)) // can I omit &a, &b and pass anonymously as a function?
            .collect(); // is there no equivalent to Haskell's zipwith?

        // small observation: removing the residual
        let x_residual = x.clone(); //clone??
        x = rmsnorm(&x);
        x = linear(&x, &state_dict.get(&format!("layer{li}.mlp_fc1")).unwrap());
        x = x.iter().map(Value::relu).collect();
        x = linear(&x, &state_dict.get(&format!("layer{li}.mlp_fc2")).unwrap());
        x = x
            .iter()
            .zip(x_residual)
            .map(|(a, b)| Value::add(&a, &b)) // can I omot &a, &b and pass anonymously?
            .collect(); // is there no equivalent to Haskell's zipwith?
    }
    // is there a way to name and return in one expression? Why can't let return the value instead of unit ()
    let logits: Vec<Rc<RefCell<Value>>> =
        linear(&x, &state_dict.get(&"lm_head".to_string()).unwrap());
    logits
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = rand::rng();
    // Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
    // This is kind of awkward, but so be it
    let names = if Path::new("input.txt").exists() {
        let mut names = String::new();
        File::open(Path::new("input.txt"))?.read_to_string(&mut names)?;
        names
    } else {
        let text = reqwest::blocking::get(
            "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
        )?
        .text()?;
        std::fs::write("input.txt", &text)?;
        text
    };

    let mut docs: Vec<&str> = names
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect(); // I don't think I really need to trim the source data, but eh
    docs.shuffle(&mut rng);
    println!("num docs: {}", docs.len());

    // Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    // unique characters in the dataset become token ids 0..n-1
    // There's a few ways to do this mapping, and I don't love this
    // I initially went weith a HashSet, converted to vec, and then sorted
    // but BTreeSet sorts on insert, which is nice
    // the issue is we need to do a full scan later to do the bidirectional tokenisation, to find the index of the character using position. But c'est la vie (it's tiny anyways)
    let uchars: BTreeSet<char> = BTreeSet::from_iter(docs.join(&"").chars()); // or docs.flatten()?
    let uchars: Vec<&char> = uchars.iter().collect();
    let BOS = uchars.len(); // token id for a special Beginning of Sequence (BOS) token
    let vocab_size = uchars.len() + 1; // total number of unique tokens, +1 is for BOS
    println!("vocab size: {}", vocab_size);

    let n_layer = 1;
    let n_embd = 16;
    let block_size = 16;
    let n_head = 4;
    let head_dim = n_embd / n_head;

    // Hacky, but I kind of love it
    let mut state_dict: HashMap<String, Matrix> = HashMap::new();
    state_dict.insert("wte".to_string(), matrix(vocab_size, n_embd));
    state_dict.insert("wpe".to_string(), matrix(block_size, n_embd));
    state_dict.insert("lm_head".to_string(), matrix(vocab_size, n_embd));

    for i in 0..n_layer {
        state_dict.insert(format!("layer{i}.attn_wq"), matrix(n_embd, n_embd));
        state_dict.insert(format!("layer{i}.attn_wk"), matrix(n_embd, n_embd));
        state_dict.insert(format!("layer{i}.attn_wv"), matrix(n_embd, n_embd));
        state_dict.insert(format!("layer{i}.attn_wo"), matrix(n_embd, n_embd));
        state_dict.insert(format!("layer{i}.mlp_fc1"), matrix(4 * n_embd, n_embd));
        state_dict.insert(format!("layer{i}.mlp_fc2"), matrix(n_embd, 4 * n_embd));
    }

    let params: Vec<&ValueRef> = state_dict.values().flatten().flatten().collect();
    println!("num params: {}", params.len());

    // Let there be Adam, the blessed optimizer and its buffers
    let (learning_rate, beta1, beta2, eps_adam) = (0.01, 0.85, 0.99, 1e-8);
    let mut m = vec![0.0; params.len()]; // first moment buffer
    let mut v = vec![0.0; params.len()]; // second moment buffer

    // Repeat in sequence
    let num_steps = 1000;
    for step in 0..num_steps {
        // Take single document, tokenize it, surround it with BOS special token on both sides
        let doc: &str = docs[step % docs.len()];
        // Don't love this code, think I can def make it more streamlined
        // let tokens = [BOS] + doc.to_string().chars().map(|ch| uchars.iter().position(|c| **c == ch)); //NEED INDEX OF element in set (HashSet?)
        // let tokens: HashMap<char, usize> = HashMap::from_iter(doc.chars().enumerate());
        let mut tokens = vec![BOS];
        tokens.extend(
            doc.chars()
                .map(|ch| uchars.iter().position(|&&c| c == ch).unwrap()),
        );
        tokens.push(BOS);
        let n = usize::min(block_size, tokens.len() - 1);

        // Forward the token sequence through the model, building up the computation graph all the way to the loss
        // for efficiency, I think I can calculate and pre-allocate the max-capacity to avoid unnecessary allocs
        let mut keys: Vec<Matrix> = vec![Vec::new(); n_layer]; // cleaner way?
        let mut values: Vec<Matrix> = vec![Vec::new(); n_layer]; // cleaner way?
        let mut losses: Vec<ValueRef> = Vec::new();
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
            let loss_t: ValueRef = Value::neg(&Value::log(&probs[target_id]));
            losses.push(loss_t);
        }
        let loss: ValueRef = Value::mul(&Value::new(1.0 / n as f64), &sum(&losses)); // final average loss over the document sequence. May yours be low.

        // Backward the loss, calculating the gradients with respect to all model parameters
        Value::backward(&loss);

        // Adam optimizer update: update the model parameters based on the corresponding gradients
        let lr_t = learning_rate * (1.0 - (step as f64) / (num_steps as f64)); // linear learning rate decay
        for (i, p) in params.iter().enumerate() {
            m[i] = beta1 * m[i] + (1.0 - beta1) * p.borrow().grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * p.borrow().grad.powi(2);
            let m_hat = m[i] / (1.0 - beta1.powi(step as i32 + 1 as i32));
            let v_hat = v[i] / (1.0 - beta2.powi(step as i32 + 1 as i32));
            p.borrow_mut().data -= lr_t * m_hat / (v_hat.powf(0.5) + eps_adam);
            p.borrow_mut().grad = 0.0;
        }
        print!(
            "step {} / {} | loss {}\r",
            step + 1,
            num_steps,
            loss.borrow().data
        ); // TODO: add format specifiers
    }

    // Inference: may the model babble back to us
    let temperature = 0.5; // in (0, 1], control the "creativity" of generated text, low to high
    println!("\n--- inference (new, hallucinated names) ---");
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
                    .map(|l| Value::truediv(l, &Value::new(temperature)))
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
        println!("sample {}: {}", sample_idx + 1, String::from_iter(sample));
    }
    Ok(())
}
