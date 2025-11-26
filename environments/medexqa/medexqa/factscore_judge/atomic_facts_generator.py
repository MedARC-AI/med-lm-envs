import json
from typing import List

from openai import AsyncOpenAI


class AtomicFactGenerator:
    """
    MedExQA-specific atomic facts generator.

    Extracts concise, checkable medical claims from an MCQA explanation that
    support the chosen option and, when useful, refute key distractors.
    Returns a Python list of strings (facts), not raw model text.
    """

    def __init__(self, async_openai_client: AsyncOpenAI | None, model_name: str = "gpt-4o-mini") -> None:
        self.client = async_openai_client
        self.model_name = model_name

    async def run(self, explanation_text: str, state: dict = None) -> List[str]:
        """
        Extract atomic facts from an MCQA explanation.
        
        Args:
            explanation_text: The explanation text to extract claims from
            state: Optional state dict for token tracking
        """
        explanation = (explanation_text or "").strip()
        if not explanation:
            return []

        primary = await self._extract_json_claims(explanation, state=state)
        if primary:
            return primary

        fallback = await self._extract_json_claims(explanation, fallback=True, state=state)
        return fallback or []

    async def _extract_json_claims(self, explanation: str, fallback: bool = False, state: dict = None) -> List[str]:
        if self.client is None:
            return []

        if not fallback:
            prompt = (
                "You are a medical expert evaluating MCQA (multiple-choice question) explanations.\n"
                "Extract atomic, checkable medical claims that: (1) justify why the correct option is right, "
                "(2) when applicable, explain why key distractors are wrong, (3) preserve medical terminology.\n\n"
                "Rules:\n"
                "- Output a strict JSON array of strings ONLY (no extra text).\n"
                "- Extract 5-7 MOST IMPORTANT claims (prioritize key medical concepts).\n"
                "- Each claim ≤ 30 words; no duplicates; no vague statements.\n"
                "- Preserve technical terms and abbreviations (e.g., 'DEXA', 'PTFE', 'AAC').\n"
                "- If no checkable medical content, return [].\n\n"
                "Few-shot examples (imitate format exactly):\n\n"
                "# Biomedical Engineering Example 1:\n"
                "Explanation: Membrane oxygenators require materials with high gas permeability for O2 and CO2 exchange. "
                "Silicone rubber, polypropylene, and Teflon are highly permeable polymers. Ceramic membranes are dense, "
                "brittle, have poor gas permeability, and can cause hemolysis.\n"
                "Claims JSON: [\n"
                "  \"Membrane oxygenators require high gas permeability for O2 and CO2 exchange.\",\n"
                "  \"Silicone rubber has excellent gas permeability and biocompatibility.\",\n"
                "  \"Polypropylene provides high gas transfer and is durable.\",\n"
                "  \"Teflon (PTFE) is chemically inert with good blood contact properties.\",\n"
                "  \"Ceramic membranes have poor gas permeability compared to polymers.\",\n"
                "  \"Ceramic membranes are brittle and can cause hemolysis.\"\n"
                "]\n\n"
                "# Biomedical Engineering Example 2:\n"
                "Explanation: Thermographic cameras detect infrared radiation emitted by objects due to temperature. "
                "All objects above absolute zero emit infrared radiation. X-rays and UV are higher-energy and not used for thermal imaging. "
                "Microwaves are used for radar, not temperature scanning.\n"
                "Claims JSON: [\n"
                "  \"Thermographic cameras detect infrared radiation from objects.\",\n"
                "  \"All objects above absolute zero emit infrared radiation.\",\n"
                "  \"Infrared is ideal for measuring surface temperatures.\",\n"
                "  \"X-rays are too high-energy for conventional thermal imaging.\",\n"
                "  \"Microwaves are used for radar applications, not thermal cameras.\"\n"
                "]\n\n"
                "# Clinical Laboratory Science Example 1:\n"
                "Explanation: Hemoglobin A1c measures average blood glucose over 2-3 months by detecting glycated hemoglobin. "
                "Fasting glucose only reflects current levels. Random glucose varies throughout the day. Oral glucose tolerance test is diagnostic but not for monitoring.\n"
                "Claims JSON: [\n"
                "  \"Hemoglobin A1c measures average blood glucose over 2-3 months.\",\n"
                "  \"A1c detects glycated hemoglobin formed by glucose binding.\",\n"
                "  \"Fasting glucose only reflects current blood glucose levels.\",\n"
                "  \"Random glucose varies throughout the day and is unreliable for averages.\",\n"
                "  \"OGTT is diagnostic but not suitable for long-term monitoring.\"\n"
                "]\n\n"
                "# Clinical Laboratory Science Example 2:\n"
                "Explanation: Gram staining differentiates bacteria by cell wall structure. Gram-positive bacteria have thick peptidoglycan walls "
                "that retain crystal violet stain. Gram-negative bacteria have thin peptidoglycan and outer membranes, appearing pink after counterstaining.\n"
                "Claims JSON: [\n"
                "  \"Gram staining differentiates bacteria by cell wall structure.\",\n"
                "  \"Gram-positive bacteria have thick peptidoglycan cell walls.\",\n"
                "  \"Thick peptidoglycan retains crystal violet stain in Gram-positive bacteria.\",\n"
                "  \"Gram-negative bacteria have thin peptidoglycan and outer membranes.\",\n"
                "  \"Gram-negative bacteria appear pink after safranin counterstaining.\"\n"
                "]\n\n"
                "# Clinical Psychology Example 1:\n"
                "Explanation: Cognitive-behavioral therapy (CBT) is first-line for generalized anxiety disorder, with strong evidence for efficacy. "
                "Psychodynamic therapy lacks robust evidence for GAD. Exposure therapy is specific to phobias. Supportive therapy alone is insufficient for GAD.\n"
                "Claims JSON: [\n"
                "  \"CBT is first-line treatment for generalized anxiety disorder.\",\n"
                "  \"CBT has strong evidence for efficacy in treating GAD.\",\n"
                "  \"Psychodynamic therapy lacks robust evidence for GAD treatment.\",\n"
                "  \"Exposure therapy is specific to phobias, not GAD.\",\n"
                "  \"Supportive therapy alone is insufficient for GAD management.\"\n"
                "]\n\n"
                "# Clinical Psychology Example 2:\n"
                "Explanation: The PHQ-9 is a validated 9-item screening tool for major depressive disorder with scores 0-27. "
                "Scores ≥10 indicate moderate depression requiring clinical evaluation. It assesses DSM-5 criteria for MDD.\n"
                "Claims JSON: [\n"
                "  \"PHQ-9 is a validated screening tool for major depressive disorder.\",\n"
                "  \"PHQ-9 contains 9 items with total scores ranging 0-27.\",\n"
                "  \"Scores ≥10 indicate moderate depression needing evaluation.\",\n"
                "  \"PHQ-9 assesses DSM-5 diagnostic criteria for MDD.\"\n"
                "]\n\n"
                "# Occupational Therapy Example 1:\n"
                "Explanation: The Barthel Index measures independence in activities of daily living (ADL) across 10 domains. "
                "Scores range 0-100, with higher scores indicating greater independence. It's reliable for tracking functional recovery post-stroke.\n"
                "Claims JSON: [\n"
                "  \"Barthel Index measures independence in activities of daily living.\",\n"
                "  \"The index assesses 10 functional domains.\",\n"
                "  \"Scores range from 0 (dependent) to 100 (independent).\",\n"
                "  \"Higher Barthel Index scores indicate greater functional independence.\",\n"
                "  \"Barthel Index is reliable for tracking post-stroke recovery.\"\n"
                "]\n\n"
                "# Occupational Therapy Example 2:\n"
                "Explanation: Adaptive utensils with built-up handles improve grip for patients with arthritis by reducing required pinch force. "
                "Weighted utensils help tremor patients. Angled utensils assist those with limited wrist mobility. Standard utensils lack these modifications.\n"
                "Claims JSON: [\n"
                "  \"Built-up handle utensils improve grip for arthritis patients.\",\n"
                "  \"Built-up handles reduce required pinch force during eating.\",\n"
                "  \"Weighted utensils help stabilize tremors during eating.\",\n"
                "  \"Angled utensils assist patients with limited wrist mobility.\",\n"
                "  \"Standard utensils lack these adaptive modifications.\"\n"
                "]\n\n"
                "# Speech Pathology Example 1:\n"
                "Explanation: Videofluoroscopic swallow study (VFSS) is the gold standard for dysphagia evaluation, visualizing all swallowing phases. "
                "It detects aspiration, penetration, and pharyngeal residue. Clinical swallow exam cannot visualize aspiration. Endoscopy misses oral phase.\n"
                "Claims JSON: [\n"
                "  \"VFSS is the gold standard for dysphagia evaluation.\",\n"
                "  \"VFSS visualizes all phases of swallowing in real-time.\",\n"
                "  \"VFSS can detect aspiration, penetration, and pharyngeal residue.\",\n"
                "  \"Clinical swallow examination cannot visualize aspiration.\",\n"
                "  \"Fiberoptic endoscopic evaluation misses the oral phase of swallowing.\"\n"
                "]\n\n"
                "# Speech Pathology Example 2:\n"
                "Explanation: The Peabody Picture Vocabulary Test (PPVT) assesses receptive vocabulary in children and adults. "
                "It requires pointing to pictures, not verbal responses, making it suitable for nonverbal individuals. Expressive language tests require speech production.\n"
                "Claims JSON: [\n"
                "  \"PPVT assesses receptive vocabulary in children and adults.\",\n"
                "  \"PPVT requires pointing to pictures, not verbal responses.\",\n"
                "  \"PPVT is suitable for assessing nonverbal individuals.\",\n"
                "  \"Expressive language tests require speech production.\"\n"
                "]\n\n"
                "Now extract claims for the explanation below.\n\n"
                f"Explanation:\n{explanation}\n\n"
                "Claims JSON:"
            )
        else:
            prompt = (
                "Extract atomic, checkable medical claims from this MCQA explanation.\n"
                "Return ONLY a JSON array of 4–10 strings; each ≤ 30 words. If none, return [].\n\n"
                f"Explanation:\n{explanation}\n\n"
                "Claims JSON:"
            )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            text = (resp.choices[0].message.content or "").strip()
            return _parse_json_list(text)
        except Exception:
            return []


def _parse_json_list(text: str) -> List[str]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = []
            for x in data:
                s = (str(x) or "").strip()
                if s:
                    out.append(s)
            # keep unique order
            seen = set()
            uniq = []
            for s in out:
                if s not in seen:
                    uniq.append(s)
                    seen.add(s)
            return uniq
        return []
    except Exception:
        # fallback: find bracketed content
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                data = json.loads(text[start : end + 1])
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
        return []





