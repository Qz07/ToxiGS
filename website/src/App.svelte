<script>
  const sections = [
    { id: "intro", label: "Introduction" },
    { id: "methods", label: "Methods" },
    { id: "contrib", label: "Contributions" },
    { id: "training", label: "Training Setup" },
    { id: "logging", label: "Logging & Evaluation" }
  ];

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };
</script>

<main>
  <header class="hero">
    <div class="hero-content">
      <h1>ToxiTIGS</h1>
      <h2>Machine Unlearning for Toxicity Suppression in GPT-2</h2>
      <p class="hero-tagline">
        Selectively unlearning toxic behaviors via IDK-style refusal alignment,
        preference optimization, and gradient-surgery-aware training.
      </p>
      <div class="hero-actions">
        <button on:click={() => scrollToSection("intro")}>Read overview</button>
        <button class="secondary" on:click={() => scrollToSection("methods")}>
          Explore methods
        </button>
      </div>
      <div class="hero-meta">
        <span>Model: GPT-2</span>
        <span>Unlearning: IDK DPO/NPO</span>
        <span>Optimization: PCGrad</span>
        <span>Infrastructure: FSDP on 2× A5000</span>
      </div>
    </div>
  </header>

  <nav class="top-nav">
    {#each sections as sec}
      <button on:click={() => scrollToSection(sec.id)}>{sec.label}</button>
    {/each}
  </nav>

  <section id="intro" class="section">
    <h2>1. Introduction</h2>
    <p>
      Large language models can reproduce unsafe or toxic continuations when
      prompted with harmful content present in their training data. In
      real-world settings like education, customer support, and public-sector
      deployments, we often need models that are robust to such prompts and that
      avoid generating harmful text.
    </p>
    <p>
      This project studies <strong>machine unlearning for toxicity suppression</strong>
      in autoregressive language models. The goal is <em>selective behavior
      change</em>: the model should suppress undesirable responses to a targeted
      <strong>forget set</strong> of prompts, while maintaining general language
      modeling performance on a benign <strong>retain set</strong>.
    </p>
    <p>
      We frame toxic suppression as <strong>behavioral unlearning via refusal
      alignment</strong>. On toxic prompts, the model learns to respond with a
      short, consistent IDK-style answer such as “I don’t know.” On retain
      prompts, it continues to train using standard next-token prediction.
    </p>
  </section>

  <section id="methods" class="section two-col">
    <div>
      <h2>2. Methods</h2>
      <h3>2.1 Problem setup and data</h3>
      <p>
        We assume a dataset of triples <code>(prompt, generation, label)</code>
        where <code>label = 1</code> marks toxic / forget prompts and
        <code>label = 0</code> marks retain prompts.
      </p>
      <ul>
        <li>
          <strong>Retain set</strong>: benign prompts where we preserve and
          improve standard language modeling behavior.
        </li>
        <li>
          <strong>Forget set</strong>: toxic prompts paired with toxic
          continuations; the model learns to replace these with an IDK-style
          refusal.
        </li>
      </ul>
      <p>
        For each forget prompt, we define a preferred completion
        <code>y_idk</code> such as “I don’t know.” This becomes the
        <em>chosen</em> response in a preference pair, with the original toxic
        completion as the <em>rejected</em> response.
      </p>

      <h3>2.2 Model and reference policy</h3>
      <p>
        We fine-tune a <strong>GPT-2</strong> causal language model
        <code>π_θ(y|x)</code>. In addition, we keep a frozen
        <strong>reference model</strong> <code>π_ref(y|x)</code> initialized
        from the same checkpoint. This reference anchors the optimization and
        stabilizes preference training.
      </p>
    </div>
    <div>
      <h3>2.3 Retain objective (masked SFT)</h3>
      <p>
        On retain examples, we apply standard masked next-token cross-entropy.
        Prompt tokens are masked out so that the loss is computed only over the
        generation portion of the sequence. This preserves general utility and
        fluency on benign inputs.
      </p>

      <h3>2.4 Forget objective (IDK preference optimization)</h3>
      <p>
        On forget examples, we use a DPO/NPO-style
        <strong>preference objective</strong> between the IDK refusal
        <code>y⁺ = y_idk</code> and the toxic completion
        <code>y⁻ = y_tox</code>. We compare relative sequence log probabilities
        under the current model and the frozen reference and optimize a
        logistic preference loss that increases the model’s preference for the
        IDK refusal, while regularizing against drifting too far from
        <code>π_ref</code>.
      </p>
    </div>
  </section>

  <section id="contrib" class="section">
    <h2>3. Key Contributions</h2>
    <div class="cards">
      <article class="card">
        <h3>Targeted unlearning via IDK refusal</h3>
        <p>
          A simple formulation that replaces toxic generations with a consistent
          IDK-style refusal using a DPO/NPO-style preference loss, while
          training normally on retain data.
        </p>
      </article>
      <article class="card">
        <h3>Conflict-aware optimization with PCGrad</h3>
        <p>
          We treat forgetting and retaining as two interacting objectives and
          apply PCGrad to reduce destructive gradient interference when their
          gradients conflict.
        </p>
      </article>
      <article class="card">
        <h3>Scalable GPT-2 training stack</h3>
        <p>
          The training pipeline uses FSDP on 2× A5000 GPUs with mixed
          precision, enabling efficient large-batch training with full
          experiment tracking.
        </p>
      </article>
    </div>
  </section>

  <section id="training" class="section two-col">
    <div>
      <h2>4. Joint training & PCGrad</h2>
      <p>
        A naïve joint objective simply sums the retain SFT loss and the forget
        preference loss. In practice, their gradients can be negatively
        aligned: aggressively improving the forget objective can harm retain
        performance.
      </p>
      <p>
        To address this, we apply <strong>PCGrad</strong> (Projected Conflicting
        Gradient). For each objective, we project away gradient components that
        conflict with the other objective, then combine the projected gradients
        to obtain the final update direction.
      </p>
      <p>
        This conflict-aware optimization better preserves retain behavior while
        still driving strong refusal alignment on forget prompts.
      </p>
    </div>

    <div>
      <h2>5. Training setup</h2>
      <ul>
        <li>
          <strong>Model sharding</strong>: Fully Sharded Data Parallel (FSDP)
          across two NVIDIA A5000 GPUs for parameter, gradient, and optimizer
          state sharding.
        </li>
        <li>
          <strong>Precision</strong>: mixed precision (fp16/bf16) with gradient
          clipping for stability.
        </li>
        <li>
          <strong>Batching</strong>: each step samples a forget batch for the
          IDK preference loss and a retain batch for masked SFT, with tunable
          weighting.
        </li>
        <li>
          <strong>Optimizer</strong>: AdamW with a cosine learning-rate
          schedule.
        </li>
      </ul>
    </div>
  </section>

  <section id="logging" class="section">
    <h2>6. Experiment logging & evaluation</h2>
    <p>
      Training is fully instrumented with <strong>Weights &amp; Biases</strong>
      and <strong>tqdm</strong> progress bars. We log retain SFT loss and token
      accuracy, forget preference loss, preference statistics (e.g., mean
      log-prob chosen vs. rejected), total loss, learning rate, and throughput.
    </p>
    <p>
      On top of these core signals, the evaluation protocol can be extended to
      track toxicity scores, perplexity on retain-only corpora, truthfulness or
      refusal ratios, and membership-inference-style diagnostics to probe how
      much toxic behavior remains.
    </p>
  </section>

  <footer class="footer">
    <p>
      ToxiTIGS &mdash; Machine Unlearning for Toxicity Suppression in GPT-2.
      Built with Svelte.
    </p>
  </footer>
</main>

<style>
  :global(body) {
    margin: 0;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
      sans-serif;
    color: #0f172a;
    background: radial-gradient(circle at top left, #0f172a, #020617);
  }

  main {
    min-height: 100vh;
    background: radial-gradient(circle at 0 0, #1e293b 0, #020617 55%);
    color: white;
  }

  .hero {
    padding: 4rem 1.5rem 3rem;
    display: flex;
    justify-content: center;
    background: radial-gradient(circle at top, #22d3ee33, #0f172a 60%);
  }

  .hero-content {
    max-width: 960px;
  }

  h1 {
    font-size: clamp(2.5rem, 4vw, 3.5rem);
    margin: 0 0 0.25rem;
    letter-spacing: 0.03em;
  }

  h2 {
    font-size: clamp(1.4rem, 2.4vw, 1.9rem);
    margin-top: 0.25rem;
    color: #e5e7eb;
    font-weight: 500;
  }

  .hero-tagline {
    margin-top: 1rem;
    max-width: 44rem;
    color: #cbd5f5;
    line-height: 1.6;
  }

  .hero-actions {
    margin-top: 1.75rem;
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  button {
    border-radius: 999px;
    border: 1px solid transparent;
    padding: 0.6rem 1.2rem;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    background: #22d3ee;
    color: #020617;
    transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s;
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.6);
  }

  button.secondary {
    background: transparent;
    color: #e5e7eb;
    border-color: #38bdf8;
    box-shadow: none;
  }

  button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.75);
    background: #06b6d4;
  }

  button.secondary:hover {
    background: #0b1120;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.7);
  }

  .hero-meta {
    margin-top: 1.75rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: #a5b4fc;
  }

  .hero-meta span {
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.4);
  }

  .top-nav {
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: rgba(15, 23, 42, 0.9);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    overflow-x: auto;
  }

  .top-nav button {
    background: transparent;
    border-radius: 999px;
    border-color: rgba(148, 163, 184, 0.7);
    color: #e5e7eb;
    box-shadow: none;
    white-space: nowrap;
  }

  .top-nav button:hover {
    background: rgba(15, 23, 42, 0.9);
    border-color: #38bdf8;
  }

  .section {
    padding: 3rem 1.5rem 2.5rem;
    max-width: 980px;
    margin: 0 auto;
  }

  .section h2 {
    color: #f9fafb;
  }

  .section h3 {
    margin-top: 1.5rem;
    color: #e5e7eb;
  }

  .section p,
  .section li {
    color: #cbd5f5;
    line-height: 1.7;
  }

  .two-col {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
    gap: 2.5rem;
    align-items: flex-start;
  }

  @media (max-width: 800px) {
    .two-col {
      grid-template-columns: minmax(0, 1fr);
    }
  }

  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
  }

  .card {
    padding: 1.4rem 1.3rem;
    border-radius: 1rem;
    background: radial-gradient(circle at top left, #1e293b, #020617);
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.85);
  }

  .card h3 {
    margin-top: 0;
    margin-bottom: 0.5rem;
  }

  .card p {
    font-size: 0.95rem;
  }

  code {
    font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo,
      Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.85em;
    padding: 0.08rem 0.35rem;
    border-radius: 0.4rem;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.4);
  }

  .footer {
    padding: 1.75rem 1.5rem 2.5rem;
    text-align: center;
    color: #9ca3af;
    border-top: 1px solid rgba(148, 163, 184, 0.24);
    background: radial-gradient(circle at bottom, #020617, #020617 55%);
  }

  .footer p {
    margin: 0;
  }
</style>
