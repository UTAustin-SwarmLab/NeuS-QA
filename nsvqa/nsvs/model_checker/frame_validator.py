import enum
import re

from nsvqa.nsvs.video.video_frame import VideoFrame


class SymbolicFilterRule(enum.Enum):
    NOT_PROPS = "not"
    AND_PROPS = "and"
    OR_PROPS = "or"

class FrameValidator:
    def __init__(
        self,
        ltl_formula: str,
        threshold_of_probability: float = 0.5,
    ):
        self.threshold_of_probability = threshold_of_probability

        ltl_formula = ltl_formula[ltl_formula.find('[') + 1:ltl_formula.rfind(']')]
        if " U " in ltl_formula:
            rule_1 = self.get_symbolic_rule_from_ltl_formula(ltl_formula.split(" U ")[0])
            rule_2 = self.get_symbolic_rule_from_ltl_formula(ltl_formula.split(" U ")[1])
            self.symbolic_verification_rule = {
                SymbolicFilterRule.AND_PROPS: rule_1[SymbolicFilterRule.AND_PROPS] + rule_2[SymbolicFilterRule.AND_PROPS],
                SymbolicFilterRule.OR_PROPS: rule_1.get(SymbolicFilterRule.OR_PROPS, []) + rule_2.get(SymbolicFilterRule.OR_PROPS, []),
                SymbolicFilterRule.NOT_PROPS: rule_1[SymbolicFilterRule.NOT_PROPS] or rule_2[SymbolicFilterRule.NOT_PROPS],
            }
        else:
            self.symbolic_verification_rule = self.get_symbolic_rule_from_ltl_formula(ltl_formula)

    def validate_frame(
        self,
        frame: VideoFrame,
    ):
        """Validate frame."""
        thresholded_objects = frame.thresholded_detected_objects(self.threshold_of_probability)
        if len(thresholded_objects) > 0:
            return self.symbolic_verification(frame)
        else:
            return False

    def symbolic_verification(self, frame: VideoFrame):
        """Symbolic verification."""
        not_props = self.symbolic_verification_rule.get(SymbolicFilterRule.NOT_PROPS)
        if not_props:
            for prop in frame.object_of_interest.keys():
                if frame.object_of_interest[prop].get_detected_probability() >= self.threshold_of_probability and prop in not_props: # detected but also in not_props
                    return False

        or_props = self.symbolic_verification_rule.get(SymbolicFilterRule.OR_PROPS)
        if or_props:
            or_group_satisfied = False
            for group in or_props:
                for prop in group:
                    if frame.object_of_interest[prop].get_detected_probability() >= self.threshold_of_probability:
                        or_group_satisfied = True
                        break
                if or_group_satisfied:
                    break
            if not or_group_satisfied:
                return False

        and_props = self.symbolic_verification_rule.get(SymbolicFilterRule.AND_PROPS)
        if and_props:
            for group in and_props:
                bad = 0
                total = 0
                for prop in group:
                    total += 1
                    if frame.object_of_interest[prop].get_detected_probability() < self.threshold_of_probability:
                        bad += 1
                if total > 2 * bad:
                    return True
        
        has_positive_props = False
        if and_props and any(and_props):
            has_positive_props = True
        if or_props and any(or_props):
            has_positive_props = True

        if not has_positive_props:
            return True

        return False

    def get_symbolic_rule_from_ltl_formula(self, ltl_formula: str) -> dict:
        symbolic_verification_rule = {}

        if "!" in ltl_formula:
            match = re.search(r'(?<!\w)!\s*(?:\((.*?)\)|([^\s\)]+))', ltl_formula)
            not_tl = (match.group(1) or match.group(2)).strip()
            symbolic_verification_rule[SymbolicFilterRule.NOT_PROPS] = not_tl
        else:
            symbolic_verification_rule[SymbolicFilterRule.NOT_PROPS] = None

        ltl_formula = re.sub(r"[!GF]", "", ltl_formula.strip())
        while ltl_formula.startswith("(") and ltl_formula.endswith(")") and ltl_formula.count("(") == ltl_formula.count(")"):
            ltl_formula = ltl_formula[1:-1].strip()

        split_and_clean_and = lambda expr: [re.sub(r"[()]", "", p).strip() for p in re.split(r"\s*&\s*", expr) if p.strip()]
        split_and_clean_or = lambda expr: [re.sub(r"[()]", "", p).strip() for p in re.split(r"\s*\|\s*", expr) if p.strip()]

        match = re.search(r'\b( U |F)\b', ltl_formula)
        if match:
            clauses = [ltl_formula[:match.start()], ltl_formula[match.start() + len(match.group(1)):]]
        else:
            clauses = [ltl_formula]

        and_props = []
        or_props = []
        for clause in clauses:
            clause = clause.strip()
            if "|" in clause:
                or_props.append(split_and_clean_or(clause))
            else:
                and_props.append(split_and_clean_and(clause))

        and_props = [[s.strip('"') for s in sublist] for sublist in and_props]
        or_props = [[s.strip('"') for s in sublist] for sublist in or_props]
        symbolic_verification_rule[SymbolicFilterRule.AND_PROPS] = and_props
        symbolic_verification_rule[SymbolicFilterRule.OR_PROPS] = or_props

        return symbolic_verification_rule
