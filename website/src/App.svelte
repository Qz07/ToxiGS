<script>
  const sections = [
    { id: "overview", label: "Overview" },
    { id: "method", label: "Method" },
    { id: "experiments", label: "Experiments" },
    { id: "results", label: "Results" },
    { id: "discussion", label: "Discussion" },
    { id: "paper-code", label: "Paper & Code" }
  ];

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };
</script>

<main>
  <header class="hero">
    <div class="hero-content">
      <h1>Detoxification via Gradient Surgery</h1>
      <h2>Conflict-aware Machine Unlearning for Safer GPT-2</h2>
      <p class="hero-tagline">
        Large language models retain harmful behaviors from toxic training data.
        We cast detoxification as a multi-task unlearning problem and apply
        gradient surgery (PCGrad) to simultaneously suppress toxicity and
        preserve language modeling ability.
      </p>
      <div class="hero-meta">
        <span>Model: GPT-2</span>
        <span>Unlearning: GradDiff &amp; idkDPO</span>
        <span>Optimization: PCGrad</span>
        <span>Dataset: ToxiGen + WikiText</span>
      </div>
      <div class="hero-authors">
        <span>Qirui Zheng, Jun-Kun Wang</span>
        <span>Halıcıoğlu Data Science Institute, UC San Diego</span>
      </div>
    </div>
  </header>

  <nav class="top-nav">
    {#each sections as sec}
      <button on:click={() => scrollToSection(sec.id)}>{sec.label}</button>
    {/each}
  </nav>

  <section id="overview" class="section">
    <h2>1. Overview</h2>
    <p>
      Large language models are trained on web-scale corpora that contain
      harmful and toxic content. After fine-tuning, these models can still
      produce toxic generations, especially under adversarial prompts. Simply
      retraining a foundation model on carefully filtered data is often
      infeasible, motivating <strong>machine unlearning</strong> as a
      post-training remedy.
    </p>
    <p>
      We study <strong>detoxification via gradient surgery</strong> for
      GPT-2-scale causal language models. The goal is to forget toxic behavior
      on a designated <strong>forget set</strong> of examples while preserving
      general language modeling ability on a <strong>retain set</strong>.
      This creates inherently conflicting optimization objectives.
    </p>
    <p>
      Our key idea is to cast unlearning as a <strong>multi-task learning</strong>
      problem with two tasks: (i) a forget objective that suppresses toxic or
      targeted behavior, and (ii) a retain objective that keeps next-token
      prediction performance high. We then apply <strong>PCGrad</strong> to
      explicitly resolve gradient conflicts between these tasks.
    </p>
  </section>

  <section id="method" class="section two-col">
    <div>
      <h2>2. Method</h2>
      <h3>2.1 Problem setting</h3>
      <p>
        We work with a causal language model <code>π_θ</code> and prompt-
        completion pairs <code>(x, y)</code>. Prompt tokens serve as context,
        and the loss is applied only on completion tokens so that we suppress
        <em>toxic continuations</em> rather than erasing prompts themselves.
      </p>
      <ul>
        <li>
          <strong>Forget set</strong> <code>D_f</code>: toxic prompts and
          continuations whose behavior we want to unlearn.
        </li>
        <li>
          <strong>Retain set</strong> <code>D_r</code>: non-toxic or
          utility-preserving data used to regularize against catastrophic
          degradation.
        </li>
      </ul>
      <p>
        The base model is a GPT-2 language model fine-tuned on a 250k-example
        corpus. All unlearning methods start from this checkpoint and are
        trained using ToxiGen toxic examples for forgetting and WikiText for
        utility evaluation.
      </p>

      <h3>2.2 Unlearning as multi-task optimization</h3>
      <p>
        Most unlearning methods decompose into a forget loss
        <code>L_f(θ)</code> on <code>D_f</code> and a retain loss
        <code>L_r(θ)</code> on <code>D_r</code>. We treat these as two
        interacting tasks with gradients <code>g_f</code> and
        <code>g_r</code>. When their inner product is negative, the objectives
        conflict, and naïve joint optimization can cause
        <strong>catastrophic collapse</strong> of model utility.
      </p>
    </div>
    <div>
      <h3>2.3 GradDiff</h3>
      <p>
        GradDiff combines gradient ascent on forget examples with standard
        next-token training on retain data. It maximizes loss on toxic targets
        so the model becomes worse at reproducing them, while minimizing loss
        on benign data to preserve utility. This objective is simple but can be
        unstable and prone to collapse.
      </p>

      <h3>2.4 idkDPO</h3>
      <p>
        idkDPO is a preference-based unlearning method. For each toxic prompt
        we construct a pair of responses: a safe "I don't know"-style
        abstention and the original toxic continuation. A DPO-style objective
        encourages the model to prefer the abstention while staying close to a
        frozen reference model. This provides a strong inductive bias toward
        safe refusals.
      </p>
    </div>
  </section>


    <section id="experiments" class="section two-col">
      <div>
        <h2>3. Experiments</h2>
        <h3>3.1 Data</h3>
        <p>
          Detoxification is evaluated on toxic language from the
          <strong>ToxiGen</strong> benchmark, which includes adversarial and
          implicit hate speech. ToxiGen provides prompts, toxic generations, and
          labels; toxic examples define <code>D_f</code> and non-toxic examples
          contribute to <code>D_r</code>.
        </p>
        <p>
          We further filter sequences to have at most 256 tokens, consistent
          with the capacity of our GPT-2 model. Utility is measured on
          <strong>WikiText</strong> word perplexity to track general language
          modeling performance.
        </p>
        <h3>3.2 Training setup</h3>
        <ul>
          <li>Base model: GPT-2 fine-tuned on ~250k examples.</li>
          <li>Hardware: 1 node with 2× NVIDIA A5000 GPUs.</li>
          <li>Distributed training: PyTorch DDP + FSDP sharding.</li>
          <li>Precision: mixed precision with gradient clipping.</li>
        </ul>
      </div>
      <div>
        <h3>3.3 Evaluation metrics</h3>
        <p>
          We evaluate along three axes that together capture detoxification
          quality and utility preservation:
        </p>
        <ul>
          <li>
            <strong>Toxicity</strong>: mean toxicity score of generations under
            a pretrained classifier (unitary/unbiased-toxic-roberta).
          </li>
          <li>
            <strong>Utility</strong>: WikiText word perplexity as a proxy for
            general language modeling ability.
          </li>
          <li>
            <strong>Membership inference</strong>: token-level NLL on member and
            non-member examples and ROC-AUC of an attack based on
            <code>-NLL</code>.
          </li>
        </ul>
      </div>
    </section>

    <section id="results" class="section">
      <h2>4. Results</h2>
      <div class="cards">
        <article class="card">
          <h3>Toxicity vs. perplexity</h3>
          <p>
            All unlearning methods reduce toxicity relative to the fine-tuned
            base model. GradDiff+PCGrad achieves the lowest toxicity score but
            suffers from very high perplexity, indicating degraded language
            modeling quality. In contrast, idkDPO and especially
            idkDPO+PCGrad reduce toxicity while maintaining low WikiText
            perplexity close to or better than the base model.
          </p>
        </article>
        <article class="card">
          <h3>Effect of PCGrad</h3>
          <p>
            Adding PCGrad consistently improves the toxicity–utility trade-off
            within each objective family. For GradDiff, PCGrad further reduces
            toxicity and somewhat improves perplexity. For idkDPO, PCGrad lowers
            toxicity and slightly improves perplexity, yielding the strongest
            overall balance.
          </p>
        </article>
        <article class="card">
          <h3>Membership inference</h3>
          <p>
            Membership inference results must be interpreted jointly with
            utility. idkDPO and idkDPO+PCGrad move ROC-AUC away from the base
            model while keeping member and non-member NLL close, suggesting more
            controlled unlearning. GradDiff-based methods show extreme NLL
            separation and ROC-AUC near zero, reflecting severe distributional
            distortion rather than desirable privacy.
          </p>
        </article>
      </div>

      <div class="result-images">
        <figure>
          <img src="performance_utility.png" alt="Table showing toxicity score versus WikiText perplexity for different unlearning methods" />
          <figcaption>
            Quantitative trade-off between toxicity and WikiText perplexity for
            the base model and all unlearning variants.
          </figcaption>
        </figure>

        <figure>
          <img src="membership_inference.png" alt="Membership inference and ROC-AUC curves comparing methods" />
          <figcaption>
            Membership inference ROC curves and NLL separation, highlighting the
            difference between GradDiff-style and idkDPO-style unlearning.
          </figcaption>
        </figure>
      </div>
    </section>

    <section id="discussion" class="section two-col">
      <div>
        <h2>5. Discussion</h2>
        <p>
          Our results support viewing language model unlearning as a
          multi-objective optimization problem. Forgetting toxic behavior and
          preserving utility induce inherently conflicting gradients, and
          conflict-aware optimization with PCGrad helps navigate this tension.
        </p>
        <p>
          For unstable objectives like GradDiff, PCGrad mitigates but does not
          eliminate collapse. For structured objectives like idkDPO, PCGrad is a
          particularly effective enhancement, improving both toxicity and
          perplexity.
        </p>
      </div>
      <div>
        <h3>Limitations & future work</h3>
        <p>
          Our study focuses on GPT-2-scale models and a detoxification setting
          derived from ToxiGen. Future work includes scaling to larger
          instruction-tuned models, extending to factual and privacy-driven
          deletion tasks, and performing more mechanistic analyses of where
          gradient conflict arises in the network.
        </p>
      </div>
    </section>

    <section id="paper-code" class="section">
      <h2>6. Paper & Code</h2>
      <p>
        You can read the full paper here:
        <a href="ToxiGS.pdf" target="_blank" rel="noreferrer">
          Download paper (PDF)
        </a>
        .
      </p>
      <p>
        Code is available at:
        <a href="https://github.com/Qz07/ToxiGS" target="_blank" rel="noreferrer">
          github.com/Qz07/ToxiGS
        </a>
        .
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
    color: #111827;
    background: #f3f4f6;
  }

  main {
    min-height: 100vh;
    background: #f3f4f6;
    color: #111827;
  }

  .hero {
    padding: 4rem 1.5rem 3rem;
    display: flex;
    justify-content: center;
    background: linear-gradient(to bottom, #ffffff, #e5e7eb);
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
    color: #374151;
    font-weight: 500;
  }

  .hero-tagline {
    margin-top: 1rem;
    max-width: 44rem;
    color: #4b5563;
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
    color: #4b5563;
  }

  .hero-meta span {
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background: #e5e7eb;
    border: 1px solid #d1d5db;
  }

  .hero-authors {
    margin-top: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    font-size: 0.85rem;
    color: #4b5563;
  }

  .top-nav {
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(209, 213, 219, 0.9);
    overflow-x: auto;
  }

  .top-nav button {
    background: transparent;
    border-radius: 999px;
    border-color: rgba(156, 163, 175, 0.8);
    color: #111827;
    box-shadow: none;
    white-space: nowrap;
  }

  .top-nav button:hover {
    background: #e5e7eb;
    border-color: #111827;
  }

  .section {
    padding: 3rem 1.5rem 2.5rem;
    max-width: 980px;
    margin: 0 auto;
  }

  .section h2 {
    color: #111827;
  }

  .section h3 {
    margin-top: 1.5rem;
    color: #111827;
  }

  .section p,
  .section li {
    color: #374151;
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
    background: #ffffff;
    border: 1px solid rgba(209, 213, 219, 0.9);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1);
  }

  .card h3 {
    margin-top: 0;
    margin-bottom: 0.5rem;
  }

  .card p {
    font-size: 0.95rem;
  }

  .result-images {
    margin-top: 2.25rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.75rem;
    align-items: flex-start;
  }

  .result-images figure {
    margin: 0;
  }

  .result-images img {
    width: 100%;
    display: block;
    border-radius: 0.9rem;
    border: 1px solid rgba(209, 213, 219, 0.9);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
  }

  .result-images figcaption {
    margin-top: 0.6rem;
    font-size: 0.85rem;
    color: #4b5563;
  }

  code {
    font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo,
      Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.85em;
    padding: 0.08rem 0.35rem;
    border-radius: 0.4rem;
    background: #e5e7eb;
    border: 1px solid #d1d5db;
  }

  .footer {
    padding: 1.75rem 1.5rem 2.5rem;
    text-align: center;
    color: #4b5563;
    border-top: 1px solid #d1d5db;
    background: #e5e7eb;
  }

  .footer p {
    margin: 0;
  }

  a {
    color: #38bdf8;
    text-decoration: none;
  }

  a:hover {
    text-decoration: underline;
  }
</style>
