# `microGPT-rs`

This is a Rust port of Andrej Karpathy's (@karpathy) [microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) training script.

The objective was to transliterate it to Rust whilst remaining as faithful to the original Python as possible. Much fun was had wrangling the borrow checker to implement the autograd graph structure for back propagation, and much gnashing of teeth was had figuring out tensor dimensions from the Python source.

Please excuse the messy comments, I'm still in the midst of cleaning it up. But it works!

To execute the training script, please run the code in `release` mode:

```shell
cargo run --release
```

Quick and dirty benchmarking suggests there's an order of magnitude difference in speed when running in `release` mode.

You should see output that looks something like the following:

```text
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.636474769018851
--- inference (new, hallucinated names) ---
sample 1: kenan
sample 2: kaman
sample 3: rakola
sample 4: rasail
sample 5: mahami
sample 6: shanal
sample 7: srin
sample 8: jeda
sample 9: alinil
sample 10: alian
sample 11: aylan
sample 12: melen
sample 13: caron
sample 14: konien
sample 15: anali
sample 16: silanun
sample 17: rori
sample 18: micain
sample 19: analen
sample 20: avilen
```

A longer write up as a blog post on https://abstractnonsense.xyz is pending.
