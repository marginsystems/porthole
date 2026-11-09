import { h } from "./ui";

type Rung = { title: string; mac: "yes" | "slow" | "no"; changes: string; body: (string | HTMLElement)[] };

const RUNGS: Rung[] = [
  {
    title: "Prompting",
    mac: "yes",
    changes: "what it focuses on, tone, format",
    body: [
      "No weights change. A system prompt and good examples are the cheapest tuning there is, and they often beat a weak fine-tune. Try it in the Chat tab, then score it in Tests.",
    ],
  },
  {
    title: "Sampling settings",
    mac: "yes",
    changes: "randomness and consistency",
    body: [
      "Temperature, top_p and top_k decide how the next word is picked from the model's probabilities. They trade variety for reliability. They never add knowledge: the model at temperature 0 is the same model, just always taking its first choice.",
    ],
  },
  {
    title: "Thinking and tools (test-time compute)",
    mac: "yes",
    changes: "how hard it works on a problem",
    body: [
      "Letting the model reason before answering, check its work, or look things up is the most reliable way to get smarter answers from the same weights. This is what porthole's harness already does with subagents. Compare the Tests score with thinking on and off.",
    ],
  },
  {
    title: "LoRA fine-tune",
    mac: "slow",
    changes: "style, format, a narrow task, a persona",
    body: [
      "You keep the 27B weights frozen and train a small add-on (a few million numbers) on your own examples. It is very good at teaching a format, a voice or a narrow skill from a few hundred to a few thousand examples. It is poor at adding new facts or general intelligence, and a bad dataset makes the model worse at everything else.",
      "mlx-vlm in porthole's venv ships a trainer. On a 27B 4-bit model expect hours, and to need most of your RAM. Not yet tried on this model:",
      h("pre", {}, "# dataset: rows with a \"messages\" column\n.venv/bin/python -m mlx_vlm.lora \\\n  --model-path Qwen3.8-27B-Uncensored-MLX/4-bit \\\n  --dataset my_examples \\\n  --lora-rank 8 --iters 600 --batch-size 1 --grad-checkpoint \\\n  --output-path adapters/my-lora.safetensors"),
      "The server can load the result with --adapter-path. Then run the Tests tab and compare against your baseline run.",
    ],
  },
  {
    title: "Steering and ablation (editing activations)",
    mac: "slow",
    changes: "one behaviour trait",
    body: [
      "This is the closest thing to “tuning neurons” that works. You record the model's internal activations on two sets of prompts (say, ones it refuses and ones it answers), take the average difference, and get a direction in its 5,120-number space. Adding or removing that direction turns the trait up or down.",
      "That is very likely how this model was made uncensored (“abliteration”): the refusal direction was projected out of the weights. It switches a behaviour off. It doesn't make the model know more or reason better, and pushing it too hard makes it incoherent.",
    ],
  },
  {
    title: "Full fine-tune",
    mac: "no",
    changes: "everything",
    body: [
      "Updating all 27 billion weights needs the weights, their gradients and optimizer state in 16/32-bit: several hundred GB of GPU memory. That is a rented multi-GPU job, not a Mac job.",
    ],
  },
  {
    title: "Hand-editing neurons",
    mac: "no",
    changes: "nothing you want",
    body: [
      "There are about 1.1 million MLP neurons and 27 billion weights. Individual neurons mostly don't mean one thing, so turning them up or down by hand almost always just breaks the model. It is a research tool (to learn what a neuron does by knocking it out), not a way to improve the model.",
    ],
  },
];

export function mountTuning(root: HTMLElement) {
  const tag = (m: Rung["mac"]) =>
    h("span", { class: "tag " + (m === "yes" ? "yes" : m === "no" ? "no" : "") },
      m === "yes" ? "easy on your Mac" : m === "slow" ? "possible on your Mac, slow" : "not on a Mac");

  root.append(h("div", { class: "grid prose", style: "max-width:860px" },
    h("div", { class: "card" },
      h("h2", {}, "“If we tune the neurons just right, could we get a really, really smart model?”"),
      h("p", {}, "In principle, yes: some setting of these 27 billion numbers is smarter than the one you have. In practice you can't find it by hand. Finding good settings for those numbers is exactly what training is, and the base Qwen took trillions of words of text and an enormous GPU cluster to get where it is."),
      h("p", {}, "Three things to know:"),
      h("ul", {},
        h("li", {}, h("b", {}, "Intelligence is spread out. "), "A skill like arithmetic lives across thousands of neurons in many layers, mixed in with other skills. There is no “smartness knob” to find."),
        h("li", {}, h("b", {}, "Size sets the ceiling. "), "A 27B model holds roughly what fits in 27B weights. Tuning changes how it uses what it has (style, focus, willingness to answer), much more than how much it has."),
        h("li", {}, h("b", {}, "Most edits make it worse. "), "Any change has to be measured. That is what the Tests tab is for: score a baseline first, then every change against it."),
      ),
      h("p", {}, "What does make it smarter on your tasks, cheapest first: better prompts, thinking mode, tools and a harness (what porthole does), a LoRA on good examples of your task, or a bigger model."),
    ),
    h("h3", {}, "Everything you can change, from cheapest to hardest"),
    h("div", { class: "ladder" }, ...RUNGS.map((r, i) =>
      h("div", { class: "rung" },
        h("div", { class: "n" }, String(i + 1)),
        h("div", {},
          h("div", {}, h("b", {}, r.title), tag(r.mac), h("span", { class: "tag" }, "changes: " + r.changes)),
          ...r.body.map((b) => (typeof b === "string" ? h("p", { style: "margin:8px 0 0" }, b) : b)),
        )))),
  ));
}
