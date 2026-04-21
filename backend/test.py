"""
tests/test_emotion_analyzer.py
--------------------------------
Unit tests for SentiFlow's emotion analysis pipeline.

These tests run WITHOUT loading any ML models — all transformer pipelines
are replaced with lightweight stubs so the suite is fast and CI-friendly.
The stubs return canned scores that let us verify every post-processing
step (blending, negation, intensifiers, keyword boost, ceiling) in
isolation from HuggingFace inference.

Run:
    pytest tests/test_emotion_analyzer.py -v
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Minimal stub factories
# ---------------------------------------------------------------------------

def _make_pipe_output(scores: dict):
    """Return a callable that mimics a transformers pipeline."""
    def _pipe(text, **kwargs):
        return [[{"label": k, "score": v} for k, v in scores.items()]]
    return _pipe


def _hartmann_stub(dominant: str, score: float = 0.80):
    """Hartmann pipeline stub with one dominant label."""
    labels = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
    out = {l: 0.03 for l in labels}
    out[dominant] = score
    return _make_pipe_output(out)


def _go_stub(dominant: str, score: float = 0.70):
    """go_emotions pipeline stub with one dominant label."""
    labels = [
        "admiration", "approval", "caring", "gratitude", "relief",
        "excitement", "optimism", "desire", "curiosity",
        "joy", "amusement", "pride", "love",
        "sadness", "grief", "disappointment", "remorse",
        "anger", "annoyance", "disgust", "fear", "nervousness",
        "surprise", "confusion", "embarrassment", "realization",
    ]
    out = {l: 0.01 for l in labels}
    out[dominant] = score
    return _make_pipe_output(out)


def _sentiment_stub(label: str = "positive", score: float = 0.90):
    return _make_pipe_output({label: score})


# ---------------------------------------------------------------------------
# Shared helper: build an analyzer with patched pipelines
# ---------------------------------------------------------------------------

def _build_analyzer(hartmann_pipe, go_pipe, sentiment_pipe):
    """
    Instantiate EmotionAnalyzer bypassing __init__ model loading,
    then inject stub pipelines directly.
    """
    from emotion_analyzer import EmotionAnalyzer
    obj = object.__new__(EmotionAnalyzer)
    obj.hartmann_pipe  = hartmann_pipe
    obj.go_pipe        = go_pipe
    obj.sentiment_pipe = sentiment_pipe
    return obj


def run(coro):
    """Run an async coroutine in tests without pytest-asyncio dependency."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# 1. Blend weights
# ===========================================================================

class TestBlending:
    def test_standard_emotion_60_40_blend(self):
        from emotion_analyzer import EmotionAnalyzer
        analyzer = _build_analyzer(None, None, None)
        h = {"joy": 0.80, "sadness": 0.10}
        g = {"joy": 0.40, "sadness": 0.20}
        blended = analyzer._blend_emotion_scores(h, g)
        expected_joy = 0.80 * 0.60 + 0.40 * 0.40
        assert abs(blended["joy"] - expected_joy) < 1e-6

    def test_trust_uses_30_70_blend(self):
        from emotion_analyzer import EmotionAnalyzer
        analyzer = _build_analyzer(None, None, None)
        h = {"trust": 0.50}
        g = {"trust": 0.80}
        blended = analyzer._blend_emotion_scores(h, g)
        expected = 0.50 * 0.30 + 0.80 * 0.70
        assert abs(blended["trust"] - expected) < 1e-6

    def test_anticipation_uses_30_70_blend(self):
        from emotion_analyzer import EmotionAnalyzer
        analyzer = _build_analyzer(None, None, None)
        h = {"anticipation": 0.60}
        g = {"anticipation": 0.90}
        blended = analyzer._blend_emotion_scores(h, g)
        expected = 0.60 * 0.30 + 0.90 * 0.70
        assert abs(blended["anticipation"] - expected) < 1e-6

    def test_missing_go_emotion_treated_as_zero(self):
        from emotion_analyzer import EmotionAnalyzer
        analyzer = _build_analyzer(None, None, None)
        h = {"anger": 0.70}
        g = {}
        blended = analyzer._blend_emotion_scores(h, g)
        assert abs(blended["anger"] - 0.70 * 0.60) < 1e-6


# ===========================================================================
# 2. Confidence ceiling
# ===========================================================================

class TestConfidenceCeiling:
    @pytest.mark.parametrize("words,ceiling", [
        (1,  0.55),
        (2,  0.55),
        (3,  0.70),
        (4,  0.70),
        (5,  0.80),
        (7,  0.80),
        (8,  0.90),
        (14, 0.90),
        (15, 0.97),
        (50, 0.97),
    ])
    def test_ceiling_thresholds(self, words, ceiling):
        from emotion_analyzer import EmotionAnalyzer
        assert EmotionAnalyzer._confidence_ceiling(words) == ceiling


# ===========================================================================
# 3. Negation detection
# ===========================================================================

class TestNegation:
    def _analyzer(self):
        return _build_analyzer(None, None, None)

    def test_not_happy_negates_joy(self):
        a = self._analyzer()
        negated = a._find_negated_emotions("I am not happy today")
        assert "joy" in negated

    def test_dont_feel_sad_negates_sadness(self):
        a = self._analyzer()
        negated = a._find_negated_emotions("I don't feel sad at all")
        assert "sadness" in negated

    def test_no_negation_returns_empty(self):
        a = self._analyzer()
        negated = a._find_negated_emotions("I feel great today")
        assert negated == []

    def test_negation_window_limited_to_5_tokens(self):
        """Negation should not apply if the keyword is >4 tokens away."""
        a = self._analyzer()
        # "not" is followed by 5 unrelated words before "happy" — outside window
        negated = a._find_negated_emotions("not one two three four five happy")
        assert "joy" not in negated


# ===========================================================================
# 4. Intensifiers and diminishers
# ===========================================================================

class TestIntensifiers:
    def _analyzer(self):
        return _build_analyzer(None, None, None)

    def test_very_amplifies_joy(self):
        a = self._analyzer()
        result = a._find_intensities("I am very happy")
        assert "joy" in result
        assert result["joy"] > 1.0

    def test_extremely_amplifies_anger(self):
        a = self._analyzer()
        result = a._find_intensities("I am extremely angry")
        assert "anger" in result
        assert result["anger"] >= 1.45

    def test_slightly_diminishes_fear(self):
        a = self._analyzer()
        result = a._find_intensities("I am slightly scared")
        assert "fear" in result
        assert result["fear"] < 1.0

    def test_no_modifier_returns_empty(self):
        a = self._analyzer()
        result = a._find_intensities("I feel okay")
        assert result == {}


# ===========================================================================
# 5. Keyword boost capping (the fix we applied)
# ===========================================================================

class TestKeywordBoost:
    def _analyzer(self):
        return _build_analyzer(None, None, None)

    def test_boost_cannot_exceed_top_model_score_plus_10pct(self):
        a = self._analyzer()
        # Anger is dominant at 0.75; joy keywords present but joy model score is 0.10
        scores = {"anger": 0.75, "joy": 0.10, "sadness": 0.05}
        # Many joy keywords to maximise boost
        text = "happy joy excited glad pleased wonderful amazing great"
        result = a._keyword_boost(text, scores)
        # Joy boosted score must not exceed 0.75 (top) + 0.10 ceiling = 0.85
        assert result["joy"] <= 0.85 + 1e-9

    def test_boost_does_not_flip_primary_emotion(self):
        a = self._analyzer()
        scores = {"anger": 0.80, "joy": 0.05}
        text = "happy joy excited glad pleased"
        result = a._keyword_boost(text, scores)
        # Joy should never exceed anger after boost
        assert result.get("anger", 0) >= result.get("joy", 0)

    def test_zero_matches_leaves_scores_unchanged(self):
        a = self._analyzer()
        scores = {"anger": 0.70, "joy": 0.10}
        result = a._keyword_boost("nothing relevant here", scores)
        assert result["anger"] == 0.70
        assert result["joy"] == 0.10


# ===========================================================================
# 6. Full async pipeline (end-to-end with stubs)
# ===========================================================================

class TestAnalyzePipeline:
    def test_joy_text_returns_joy_primary(self):
        analyzer = _build_analyzer(
            _hartmann_stub("joy", 0.85),
            _go_stub("joy", 0.75),
            _sentiment_stub("positive", 0.92),
        )
        result = run(analyzer.analyze("I am so happy and excited about this!"))
        assert result["primary_emotion"] == "joy"
        assert result["sentiment"] == "positive"

    def test_anger_text_returns_anger_primary(self):
        analyzer = _build_analyzer(
            _hartmann_stub("anger", 0.88),
            _go_stub("anger", 0.80),
            _sentiment_stub("negative", 0.85),
        )
        result = run(analyzer.analyze("This is absolutely outrageous and unacceptable!"))
        assert result["primary_emotion"] == "anger"

    def test_short_text_score_capped_by_ceiling(self):
        analyzer = _build_analyzer(
            _hartmann_stub("joy", 0.99),
            _go_stub("joy", 0.99),
            _sentiment_stub("positive", 0.99),
        )
        result = run(analyzer.analyze("Hi"))   # 1 word → ceiling 0.55 → max 55%
        assert result["emotion_score"] <= 55.0

    def test_result_contains_required_keys(self):
        analyzer = _build_analyzer(
            _hartmann_stub("sadness", 0.70),
            _go_stub("sadness", 0.60),
            _sentiment_stub("negative", 0.80),
        )
        result = run(analyzer.analyze("I feel really down today."))
        required = {
            "primary_emotion", "emotion_score", "sentiment", "sentiment_score",
            "emotion_scores", "top_3_emotions", "emoji", "color",
        }
        assert required.issubset(result.keys())

    def test_all_eight_emotions_present_in_scores(self):
        from emotion_analyzer import Emotion
        analyzer = _build_analyzer(
            _hartmann_stub("joy", 0.70),
            _go_stub("joy", 0.60),
            _sentiment_stub("positive", 0.85),
        )
        result = run(analyzer.analyze("Feeling good today!"))
        for emotion in Emotion:
            assert emotion.value in result["emotion_scores"], \
                f"Missing emotion in output: {emotion.value}"

    def test_negation_lowers_joy_score(self):
        """'not happy' should produce a lower joy score than 'very happy'."""
        analyzer_neg = _build_analyzer(
            _hartmann_stub("joy", 0.80),
            _go_stub("joy", 0.70),
            _sentiment_stub("negative", 0.70),
        )
        analyzer_pos = _build_analyzer(
            _hartmann_stub("joy", 0.80),
            _go_stub("joy", 0.70),
            _sentiment_stub("positive", 0.90),
        )
        neg_result = run(analyzer_neg.analyze("I am not happy at all."))
        pos_result = run(analyzer_pos.analyze("I am very happy today!"))
        neg_joy = neg_result["emotion_scores"]["joy"]
        pos_joy = pos_result["emotion_scores"]["joy"]
        assert neg_joy < pos_joy

    def test_empty_text_handled_gracefully(self):
        analyzer = _build_analyzer(
            _hartmann_stub("joy", 0.50),
            _go_stub("joy", 0.50),
            _sentiment_stub("neutral", 0.60),
        )
        # Should not raise even on minimal input after preprocessing
        result = run(analyzer.analyze("ok"))
        assert "primary_emotion" in result

    def test_very_long_text_truncated_to_last_3_sentences(self):
        """Preprocessing keeps last 3 sentences — models should still return a result."""
        analyzer = _build_analyzer(
            _hartmann_stub("fear", 0.75),
            _go_stub("fear", 0.65),
            _sentiment_stub("negative", 0.80),
        )
        long_text = ". ".join([f"Sentence {i} is here" for i in range(20)])
        long_text += ". I am terrified."
        result = run(analyzer.analyze(long_text))
        assert "primary_emotion" in result


# ===========================================================================
# 7. Preprocessing
# ===========================================================================

class TestPreprocessing:
    def _analyzer(self):
        return _build_analyzer(None, None, None)

    def test_html_tags_stripped(self):
        a = self._analyzer()
        result = a._preprocess("<b>Hello</b> world")
        assert "<b>" not in result
        assert "Hello" in result

    def test_urls_stripped(self):
        a = self._analyzer()
        result = a._preprocess("Check https://example.com for details")
        assert "https://" not in result

    def test_email_stripped(self):
        a = self._analyzer()
        result = a._preprocess("Contact me at user@example.com")
        assert "@" not in result

    def test_context_window_keeps_last_3_sentences(self):
        a = self._analyzer()
        sentences = [f"Sentence {i}." for i in range(10)]
        text = " ".join(sentences)
        result = a._preprocess(text)
        assert "Sentence 9" in result
        assert "Sentence 0" not in result


# ===========================================================================
# 8. Formality and clarity (main.py helpers — imported directly)
# ===========================================================================

class TestHeuristics:
    def test_formal_text_scores_above_50(self):
        from main import calculate_formality
        score = calculate_formality("Dear Sir, I would be grateful if you could kindly review the attached document.")
        assert score > 50

    def test_informal_text_scores_below_50(self):
        from main import calculate_formality
        score = calculate_formality("hey yeah gonna lol btw omg ngl idk ur")
        assert score < 50

    def test_short_sentence_clarity_reasonable(self):
        from main import calculate_clarity
        score = calculate_clarity("The meeting is at noon.")
        assert score > 60

    def test_long_complex_sentence_lower_clarity(self):
        from main import calculate_clarity
        score = calculate_clarity(
            "The aforementioned multifaceted organisational restructuring initiative "
            "necessitates a comprehensive reevaluation of the interdepartmental "
            "communication frameworks and their concomitant implications."
        )
        assert score < 70

    def test_clarity_never_negative(self):
        from main import calculate_clarity
        score = calculate_clarity("supercalifragilisticexpialidocious " * 20)
        assert score >= 0


# ===========================================================================
# 9. Suggestion generation
# ===========================================================================

class TestSuggestions:
    def test_anger_gets_tone_suggestion(self):
        from main import generate_suggestions
        suggestions = generate_suggestions("anger", 60, 70, 10)
        assert any("frustration" in s.lower() or "tone" in s.lower() for s in suggestions)

    def test_low_formality_gets_suggestion(self):
        from main import generate_suggestions
        suggestions = generate_suggestions("joy", 20, 70, 10)
        assert any("formal" in s.lower() for s in suggestions)

    def test_low_clarity_gets_suggestion(self):
        from main import generate_suggestions
        suggestions = generate_suggestions("joy", 60, 30, 10)
        assert any("clarity" in s.lower() or "shorter" in s.lower() for s in suggestions)

    def test_long_message_gets_length_suggestion(self):
        from main import generate_suggestions
        suggestions = generate_suggestions("joy", 60, 70, 200)
        assert any("lengthy" in s.lower() or "long" in s.lower() for s in suggestions)

    def test_max_3_suggestions_returned(self):
        from main import generate_suggestions
        # Trigger all conditions simultaneously
        suggestions = generate_suggestions("anger", 10, 10, 200)
        assert len(suggestions) <= 3