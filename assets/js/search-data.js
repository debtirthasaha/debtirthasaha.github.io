// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "Shipped, and in progress.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "post-a-haiku-vlm-sft-did-the-work-kto-collapsed-at-λ-1-0",
        
          title: "A haiku VLM: SFT did the work, KTO collapsed at λ=1.0",
        
        description: "A LLaVA-pattern VLM that writes a 5-7-5 haiku for a ukiyo-e woodblock print. SigLIP (frozen) + trained projector + Qwen2.5-3B (LoRA), on 3,913 Met Museum prints, in English and Japanese. SFT delivered ~95% of the lift; preference optimization only helped where the chosen/rejected gap was real; KTO collapsed at its default λ_U.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/ukiyoe-haiku-vlm/";
          
        },
      },{id: "post-a-7b-math-fine-tune-on-8-h100-sft-6-4-dpo-0-6",
        
          title: "A 7B math fine-tune on 8× H100: SFT +6.4, DPO +0.6",
        
        description: "Two-stage LoRA (SFT then DPO) on DeepSeek-R1-Distill-Qwen-7B, end-to-end on a single 8× H100 pod for ~$93. SFT lifted four math benchmarks by +6.4 pp average; DPO at conservative defaults moved nothing, and the training-time reward margin predicted it.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/math-slm/";
          
        },
      },{id: "post-eight-a100s-61-and-124m-parameters",
        
          title: "Eight A100s, $61, and 124M parameters",
        
        description: "Full reproduction of GPT-2 124M on rented multi-GPU hardware. Val loss 3.40 vs OpenAI&#39;s 3.29 (97% match), HellaSwag 27% vs 29.45%, in 2.5 hours of training.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/gpt2-124m/";
          
        },
      },{id: "post-bpe-from-scratch-and-why-your-llm-can-39-t-count-l-39-s",
        
          title: "BPE from scratch, and why your LLM can&#39;t count L&#39;s",
        
        description: "Byte-pair encoding implemented in pure Python. Plus SolidGoldMagikarp, the encode/decode asymmetry, and a list of LLM weirdness all caused by the tokenizer.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/bpe-tokenizer/";
          
        },
      },{id: "post-birkhoff-in-8-7-kb",
        
          title: "Birkhoff in 8.7 KB",
        
        description: "An 8.71 KB prompt for SAIR&#39;s equational-theories competition (Tao + Davis, follow-up to Honda-Murakami-Zhang 2025). Replace free-form LLM reasoning with a 9-magma Birkhoff-sound decision procedure. A 31B model running this prompt beat a 120B one on the hardest set.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/birkhoff-in-8kb/";
          
        },
      },{id: "post-tiny-shakespeare-tiny-gpt",
        
          title: "Tiny Shakespeare, tiny GPT",
        
        description: "A 1.83M-parameter decoder-only transformer trained on 1MB of Shakespeare. Architecture is identical to GPT-2, just smaller.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/tiny-shakespeare-tiny-gpt/";
          
        },
      },{id: "post-makemore-from-counting-bigrams-to-a-wavenet",
        
          title: "makemore: from counting bigrams to a WaveNet",
        
        description: "Five character-level language models trained on 32K baby names. Bigram → MLP → BatchNorm → manual backprop → hierarchical fusion.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/makemore/";
          
        },
      },{id: "post-micrograd-a-scalar-valued-autograd-engine",
        
          title: "micrograd: a scalar-valued autograd engine",
        
        description: "A 150-line autograd engine that supports +, *, **, tanh, exp, and a tiny MLP on top.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/building-micrograd/";
          
        },
      },{id: "post-a-transformer-that-reads-c-and-writes-python",
        
          title: "A transformer that reads C++ and writes Python",
        
        description: "16.4M-parameter encoder-decoder transformer for C++ → Python code translation, trained on XLCoST on a 4 GB GPU. val_loss 2.0474.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cpp-to-python-transformer/";
          
        },
      },{id: "projects-equational-theories-cheatsheet",
          title: 'Equational theories cheatsheet',
          description: "An 8.71 KB Birkhoff-sound prompt for SAIR&#39;s equational-theories competition (Tao + Davis). A 31B model running this prompt beat a 120B one on the hardest set.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_equational/";
            },},{id: "projects-gpt-2-124m-reproduction",
          title: 'GPT-2 (124M) reproduction',
          description: "Full Karpathy-style reproduction. 8× A100, 19073 steps over 10B FineWeb-Edu tokens, $61, val loss 3.40 (97% of OpenAI baseline).",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_gpt2/";
            },},{id: "projects-c-python-transformer",
          title: 'C++ → Python transformer',
          description: "16.4M-parameter encoder-decoder for code translation, trained on XLCoST on a GTX 1650. val_loss 2.0474.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_transformer/";
            },},{id: "projects-bpe-tokenizer",
          title: 'BPE tokenizer',
          description: "Pure-Python byte-pair encoding, plus a deep dive on why tokenization makes LLMs weird (SolidGoldMagikarp, spelling, arithmetic).",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_bpe/";
            },},{id: "projects-tiny-shakespeare-gpt",
          title: 'Tiny Shakespeare GPT',
          description: "1.83M-parameter character-level decoder transformer on Tiny Shakespeare. Same architecture as GPT-2, scaled down.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_tiny_gpt/";
            },},{id: "projects-makemore",
          title: 'makemore',
          description: "Five character-level language models on 32K baby names — bigram counts, MLP, BatchNorm, manual backprop, WaveNet-style.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_makemore/";
            },},{id: "projects-micrograd",
          title: 'micrograd',
          description: "Scalar-valued autograd engine in ~150 lines of pure Python. Supports +, *, **, tanh, exp, plus a tiny MLP.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_micrograd/";
            },},{id: "projects-math-slm-sft-dpo",
          title: 'Math SLM (SFT + DPO)',
          description: "Two-stage LoRA on DeepSeek-R1-Distill-Qwen-7B. SFT +6.4 pp across four math benchmarks; DPO a config-bottlenecked no-op. End-to-end on 8× H100 for ~$93.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_math_slm/";
            },},{id: "projects-ukiyo-e-haiku-vlm",
          title: 'Ukiyo-e Haiku VLM',
          description: "LLaVA-pattern VLM that writes a haiku for a ukiyo-e print. SigLIP + trained projector + Qwen2.5-3B LoRA, in English and Japanese. SFT did ~95% of the lift; KTO collapsed at default λ_U.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_ukiyoe_vlm/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%64%65%62%74%69%72%74%68%61%73%61%68%61%31@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/debtirthasaha", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
