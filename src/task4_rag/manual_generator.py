import os
from pathlib import Path
from typing import List, Dict

class ManualGenerator:
    """Generates synthetic maintenance manuals for different industrial defects."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_manuals(self) -> List[str]:
        """Creates markdown SOP manuals and returns their file paths."""
        manuals = {
            "scratch_repair_sop.md": self._scratch_sop(),
            "crack_welding_sop.md": self._crack_sop(),
            "corrosion_treatment_sop.md": self._corrosion_sop(),
            "dimensional_flaw_sop.md": self._dimensional_sop()
        }

        filepaths = []
        for filename, content in manuals.items():
            path = self.output_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            filepaths.append(str(path))
        
        return filepaths

    def _scratch_sop(self) -> str:
        return """# Standard Operating Procedure: Surface Scratch Repair
## 1. Safety Precautions
- **PPE Required**: Safety glasses, nitrile gloves, dust mask.
- **LOTO**: Not strictly required if component is detached, but verify zero energy state if on-line.

## 2. Required Tools & Materials
- 400-grit and 800-grit diamond polishing paste
- Microfiber cloths
- Isopropyl alcohol (99%)
- Orbital buffer

## 3. Repair Steps
1. Clean the scratched area thoroughly with isopropyl alcohol.
2. Apply 400-grit diamond paste to the orbital buffer pad.
3. Buff the scratch in the direction of the metal grain for 3-5 minutes.
4. Wipe clean and inspect. If scratch is still visible, repeat step 3.
5. Apply 800-grit paste and buff for a final polish to match surface finish.

## 4. Re-commissioning Checklist
- [ ] Visual inspection confirms scratch is removed or within acceptable tolerance (Ra < 0.8μm).
- [ ] Surface is completely cleaned of polishing residue.
"""

    def _crack_sop(self) -> str:
        return """# Standard Operating Procedure: Structural Crack Welding
## 1. Safety Precautions
- **PPE Required**: Auto-darkening welding helmet, heavy leather gloves, flame-resistant jacket, respirator.
- **LOTO**: Strict OSHA Lock-Out/Tag-Out required. Machine must be isolated from main power.

## 2. Required Tools & Materials
- GTAW (TIG) Welder
- ER70S-6 filler rod
- Angle grinder with abrasive disc
- Dye penetrant inspection kit
- Argon gas cylinder

## 3. Repair Steps
1. Stop line immediately. Execute LOTO on the affected machine.
2. Using the angle grinder, gouge out the crack to form a U-groove, extending 0.5 inches past the visible ends of the crack.
3. Clean the groove with acetone.
4. Set GTAW welder to 150A (DCEN) and establish Argon shielding gas flow at 15 CFH.
5. Weld the groove using ER70S-6 filler rod, ensuring complete penetration.
6. Grind the weld bead flush with the base metal.

## 4. Re-commissioning Checklist
- [ ] Dye penetrant test confirms no surface-breaking defects in the weld.
- [ ] LOTO tags removed by authorized personnel.
- [ ] Run equipment at 50% load for 10 minutes to verify integrity.
"""

    def _corrosion_sop(self) -> str:
        return """# Standard Operating Procedure: Corrosion & Oxidation Treatment
## 1. Safety Precautions
- **PPE Required**: Chemical splash goggles, acid-resistant apron, heavy neoprene gloves, face shield.
- **LOTO**: Ensure system is depressurized and drained of all fluids before treatment.

## 2. Required Tools & Materials
- 20% Nitric acid passivation solution
- Wire brush (stainless steel)
- Distilled water
- pH testing strips

## 3. Repair Steps
1. Manually scrub the heavy corrosion and pitting using the wire brush to remove loose scale.
2. Apply the 20% Nitric acid passivation solution to the affected area.
3. Allow the solution to dwell for 20-30 minutes, depending on ambient temperature.
4. Thoroughly rinse the area with distilled water.
5. Test the runoff water with pH strips to ensure it has returned to neutral (pH 7).

## 4. Re-commissioning Checklist
- [ ] Visual inspection confirms removal of active red/orange oxidation.
- [ ] Runoff pH is neutral.
- [ ] Apply preventative anti-corrosion spray if area is exposed to moisture.
"""

    def _dimensional_sop(self) -> str:
        return """# Standard Operating Procedure: Dimensional Flaw Remediation
## 1. Safety Precautions
- **PPE Required**: Steel-toe boots, safety glasses, hearing protection.
- **LOTO**: LOTO required on CNC machining center during setup.

## 2. Required Tools & Materials
- CNC Milling Machine
- Carbide end mills
- Digital calipers and micrometers
- CMM (Coordinate Measuring Machine)

## 3. Repair Steps
1. Measure the deformed edge or chipped corner using digital calipers to determine the extent of material loss.
2. If the flaw exceeds ±0.05mm tolerance, the component must be remachined or scrapped.
3. Secure the component in the CNC milling fixture.
4. Run the designated cleanup toolpath (Program #C-804) to shave the edge back to the next undersized tolerance tier.
5. Deburr the freshly machined edges.

## 4. Re-commissioning Checklist
- [ ] CMM inspection confirms component meets the revised dimensional specifications.
- [ ] Component is re-stamped with the undersize modification code.
"""
