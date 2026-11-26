"""Death Clock Actuarial Estimation Tool using Gemini API and Epidemiological Baseline Models.

Computes comprehensive life expectancy projections, biological age offsets,
countdown telemetry (remaining days, hours, and seconds), actionable longevity
interventions, and optional generative clinical-actuarial synthesis via Gemini.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
import os
from typing import TYPE_CHECKING, Any

import httpx

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.death_clock")

# Default Google Gemini Cloud API Key fallback for cloud synthesis
DEFAULT_GEMINI_KEY = ""

# Actuarial baseline lifespans by country (WHO Global Health Observatory 2024 data)
ACTUARIAL_COUNTRY_BASELINES: dict[str, dict[str, float]] = {
    "india": {"male": 70.4, "female": 73.6, "average": 72.0},
    "united states": {"male": 76.3, "female": 81.4, "average": 78.8},
    "usa": {"male": 76.3, "female": 81.4, "average": 78.8},
    "japan": {"male": 81.5, "female": 87.6, "average": 84.6},
    "united kingdom": {"male": 79.4, "female": 83.1, "average": 81.2},
    "uk": {"male": 79.4, "female": 83.1, "average": 81.2},
    "germany": {"male": 78.7, "female": 83.5, "average": 81.1},
    "canada": {"male": 80.6, "female": 84.7, "average": 82.6},
    "australia": {"male": 81.3, "female": 85.4, "average": 83.3},
    "default": {"male": 71.0, "female": 74.5, "average": 72.8},
}


def _calculate_actuarial_baseline(country: str, gender: str) -> float:
    """Determine baseline life expectancy using country and gender tables.

    Args:
        country: Country name or location string.
        gender: Gender string ('male', 'female', or other).

    Returns:
        Baseline actuarial lifespan in years.
    """
    country_clean = (country or "default").strip().lower()
    gender_clean = (gender or "male").strip().lower()

    table = ACTUARIAL_COUNTRY_BASELINES.get(country_clean, ACTUARIAL_COUNTRY_BASELINES["default"])
    if "female" in gender_clean or "woman" in gender_clean:
        return table.get("female", 74.5)
    if "male" in gender_clean or "man" in gender_clean:
        return table.get("male", 71.0)
    return table.get("average", 72.8)


def _compute_lifestyle_delta(
    smoking: str,
    exercise_days_per_week: int,
    diet_quality: str,
    sleep_hours: float,
    stress_level: str,
    alcohol: str,
    bmi: float | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute actuarial year adjustments based on epidemiological risk markers.

    Args:
        smoking: Smoking status ('never', 'occasional', 'heavy').
        exercise_days_per_week: Number of exercise/workout days per week (0-7).
        diet_quality: Diet nutrition rating ('poor', 'average', 'healthy', 'exceptional').
        sleep_hours: Average sleep duration in hours per night.
        stress_level: Self-reported stress level ('low', 'moderate', 'high', 'extreme').
        alcohol: Alcohol consumption status ('none', 'moderate', 'heavy').
        bmi: Optional body mass index value.

    Returns:
        Tuple of total year delta (positive or negative) and itemized impacts list.
    """
    delta: float = 0.0
    impacts: list[dict[str, Any]] = []

    # Smoking impact
    smk = (smoking or "never").strip().lower()
    if "heavy" in smk:
        d = -10.5
        impacts.append({"factor": "Smoking (Heavy)", "delta_years": d, "category": "hazard"})
    elif "occasional" in smk or "vape" in smk:
        d = -4.5
        impacts.append({"factor": "Smoking / Vaping (Occasional)", "delta_years": d, "category": "hazard"})
    else:
        d = 0.0
        impacts.append({"factor": "Non-smoker status", "delta_years": d, "category": "neutral"})
    delta += d

    # Exercise impact
    ex = max(0, min(7, exercise_days_per_week))
    if ex >= 5:
        d = 4.5
        impacts.append({"factor": f"High Physical Activity ({ex} days/wk)", "delta_years": d, "category": "enhancer"})
    elif ex >= 3:
        d = 3.2
        impacts.append({"factor": f"Regular Cardio & Resistance ({ex} days/wk)", "delta_years": d, "category": "enhancer"})
    elif ex >= 1:
        d = 1.0
        impacts.append({"factor": f"Light Movement ({ex} days/wk)", "delta_years": d, "category": "enhancer"})
    else:
        d = -3.5
        impacts.append({"factor": "Sedentary Lifestyle (0 days/wk)", "delta_years": d, "category": "hazard"})
    delta += d

    # Diet impact
    dt = (diet_quality or "average").strip().lower()
    if "exceptional" in dt or "mediterranean" in dt or "optimal" in dt:
        d = 4.0
        impacts.append({"factor": "High Whole-Food & Plant-Rich Diet", "delta_years": d, "category": "enhancer"})
    elif "healthy" in dt or "clean" in dt or "good" in dt:
        d = 2.5
        impacts.append({"factor": "Balanced Nutrient-Dense Diet", "delta_years": d, "category": "enhancer"})
    elif "poor" in dt or "junk" in dt or "fast" in dt:
        d = -3.5
        impacts.append({"factor": "Ultra-Processed & High-Sugar Diet", "delta_years": d, "category": "hazard"})
    else:
        d = 0.0
        impacts.append({"factor": "Average Standard Diet", "delta_years": d, "category": "neutral"})
    delta += d

    # Sleep impact
    sl = float(sleep_hours)
    if 7.0 <= sl <= 8.5:
        d = 1.8
        impacts.append({"factor": f"Optimal Sleep Hygiene ({sl:.1f} hrs/night)", "delta_years": d, "category": "enhancer"})
    elif sl < 6.0:
        d = -2.8
        impacts.append({"factor": f"Chronic Sleep Deprivation (<6 hrs, {sl:.1f} hrs)", "delta_years": d, "category": "hazard"})
    elif sl > 9.5:
        d = -1.2
        impacts.append({"factor": f"Excess Hypersomnia (>9.5 hrs, {sl:.1f} hrs)", "delta_years": d, "category": "hazard"})
    else:
        d = 0.5
        impacts.append({"factor": f"Acceptable Sleep Duration ({sl:.1f} hrs)", "delta_years": d, "category": "enhancer"})
    delta += d

    # Stress impact
    st = (stress_level or "moderate").strip().lower()
    if "low" in st or "calm" in st or "mindful" in st:
        d = 2.0
        impacts.append({"factor": "Low Chronic Cortisol & Stress Resiliency", "delta_years": d, "category": "enhancer"})
    elif "extreme" in st or "severe" in st or "burnout" in st:
        d = -5.0
        impacts.append({"factor": "Severe Allostatic Load & Chronic Burnout", "delta_years": d, "category": "hazard"})
    elif "high" in st:
        d = -2.5
        impacts.append({"factor": "Elevated Occupational / Life Stress", "delta_years": d, "category": "hazard"})
    else:
        d = 0.0
        impacts.append({"factor": "Moderate Coping & Daily Stress", "delta_years": d, "category": "neutral"})
    delta += d

    # Alcohol impact
    alc = (alcohol or "moderate").strip().lower()
    if "heavy" in alc or "binge" in alc:
        d = -6.5
        impacts.append({"factor": "High Alcohol Intake", "delta_years": d, "category": "hazard"})
    elif "none" in alc or "sober" in alc or "abstinent" in alc:
        d = 1.0
        impacts.append({"factor": "Alcohol Abstinence", "delta_years": d, "category": "enhancer"})
    else:
        d = 0.0
        impacts.append({"factor": "Moderate Occasional Alcohol", "delta_years": d, "category": "neutral"})
    delta += d

    # BMI impact (if provided)
    if bmi is not None and bmi > 0:
        if 18.5 <= bmi <= 24.9:
            d = 1.2
            impacts.append({"factor": f"Optimal BMI ({bmi:.1f})", "delta_years": d, "category": "enhancer"})
        elif 25.0 <= bmi <= 29.9:
            d = -1.0
            impacts.append({"factor": f"Overweight BMI Range ({bmi:.1f})", "delta_years": d, "category": "hazard"})
        elif bmi >= 30.0:
            d = -4.5
            impacts.append({"factor": f"Class 1+ Obesity BMI Range ({bmi:.1f})", "delta_years": d, "category": "hazard"})
        else:
            d = -1.8
            impacts.append({"factor": f"Underweight BMI Range ({bmi:.1f})", "delta_years": d, "category": "hazard"})
        delta += d

    return delta, impacts


async def _query_gemini_actuarial_synthesis(
    age: int,
    gender: str,
    country: str,
    estimated_lifespan: float,
    years_remaining: float,
    impacts: list[dict[str, Any]],
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Request structured clinical actuarial longevity insights from Gemini Cloud API.

    Args:
        age: Current user age in years.
        gender: Gender string.
        country: Country string.
        estimated_lifespan: Calculated life expectancy in years.
        years_remaining: Calculated remaining years.
        impacts: List of itemized factor impacts.
        api_key: Optional Gemini API key override.

    Returns:
        Structured dictionary containing clinical insights or None on error/timeout.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_GEMINI_KEY
    if not key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    prompt = (
        f"You are a clinical actuarial intelligence agent. Analyze this subject's longevity telemetry:\n"
        f"- Age: {age}, Gender: {gender}, Country: {country}\n"
        f"- Projected Lifespan: {estimated_lifespan:.1f} years (Remaining: {years_remaining:.1f} years)\n"
        f"- Lifestyle Markers: {json.dumps(impacts, ensure_ascii=False)}\n\n"
        f"Return a strict JSON object with these exact keys:\n"
        f'{{"longevity_score": <int 1-100>, "biological_age_offset": <float offset e.g. -2.5 or +3.1>, '
        f'"top_recommendations": [<string action 1>, <string action 2>, <string action 3>], '
        f'"actuarial_insight": "<2-3 sentence scientifically rigorous, motivating and witty synthesis>"}}'
    )

    body = {
        "model": "gemini-2.0-flash",
        "messages": [
            {"role": "system", "content": "You are an actuarial epidemiologist. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=4.5) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            logger.warning("gemini_actuarial_failed", status=resp.status_code)
    except Exception as exc:
        logger.warning("gemini_actuarial_exception", error=str(exc))
    return None


async def estimate_death_clock(
    age: int = 20,
    gender: str = "male",
    country: str = "India",
    smoking: str = "never",
    exercise_days_per_week: int = 3,
    diet_quality: str = "average",
    sleep_hours: float = 7.5,
    stress_level: str = "moderate",
    alcohol: str = "moderate",
    bmi: float | None = None,
    use_gemini: bool = True,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Calculate mortality timeline, actuarial life expectancy, and longevity countdown.

    Combines WHO country baseline statistics, epidemiological lifestyle regression
    coefficients, and Google Gemini Cloud synthesis into a comprehensive death clock projection.

    Args:
        age: Subject's current age in years (default 20).
        gender: Subject's biological sex or gender ('male', 'female', etc.).
        country: Country of residence (default 'India').
        smoking: Smoking status ('never', 'occasional', 'heavy').
        exercise_days_per_week: Frequency of cardio/strength workouts per week (0-7).
        diet_quality: Nutritional quality ('poor', 'average', 'healthy', 'exceptional').
        sleep_hours: Average sleep duration in hours (e.g. 7.5).
        stress_level: Chronic stress load ('low', 'moderate', 'high', 'extreme').
        alcohol: Alcohol consumption status ('none', 'moderate', 'heavy').
        bmi: Optional body mass index value.
        use_gemini: Whether to query Gemini Cloud API for generative clinical insights.
        api_key: Optional Gemini API key override.

    Returns:
        Structured dictionary containing projected lifespan, remaining time countdown,
        lifestyle factor breakdown, and longevity recommendations.
    """
    clean_age = max(1, min(115, int(age)))
    baseline_lifespan = _calculate_actuarial_baseline(country, gender)
    delta_years, impacts = _compute_lifestyle_delta(
        smoking=smoking,
        exercise_days_per_week=exercise_days_per_week,
        diet_quality=diet_quality,
        sleep_hours=sleep_hours,
        stress_level=stress_level,
        alcohol=alcohol,
        bmi=bmi,
    )

    # Calculate final lifespan clamped realistically between age+0.5 and 115
    projected_lifespan = max(float(clean_age) + 0.5, min(115.0, baseline_lifespan + delta_years))
    years_remaining = max(0.5, projected_lifespan - float(clean_age))

    # Compute high-precision countdown
    now = datetime.now()
    days_remaining = int(years_remaining * 365.2425)
    hours_remaining = days_remaining * 24
    seconds_remaining = hours_remaining * 3600

    estimated_target_date = now + timedelta(days=days_remaining)
    target_year = estimated_target_date.year

    # Generative Gemini Synthesis (if enabled)
    gemini_data: dict[str, Any] | None = None
    if use_gemini:
        gemini_data = await _query_gemini_actuarial_synthesis(
            age=clean_age,
            gender=gender,
            country=country,
            estimated_lifespan=projected_lifespan,
            years_remaining=years_remaining,
            impacts=impacts,
            api_key=api_key,
        )

    # Derive biological age offset and score
    bio_age_offset = delta_years * -0.7
    bio_age = max(1.0, float(clean_age) + bio_age_offset)

    enhancers = [item for item in impacts if item["category"] == "enhancer"]
    hazards = [item for item in impacts if item["category"] == "hazard"]

    # Actionable suggestions
    recommendations: list[str] = []
    if gemini_data and "top_recommendations" in gemini_data:
        recommendations = gemini_data["top_recommendations"]
    else:
        if exercise_days_per_week < 4:
            recommendations.append("Increase zone 2 cardio & strength training to 4+ days/week (+3.2 years).")
        if sleep_hours < 7.0:
            recommendations.append("Optimize sleep hygiene to achieve 7.5-8 hours nightly (+1.8 years).")
        if "heavy" in smoking or "occasional" in smoking:
            recommendations.append("Smoking cessation immediately halts vascular decay (+4.5 to +10.5 years).")
        if "poor" in diet_quality:
            recommendations.append("Shift to whole-food Mediterranean diet pattern (+3.5 years).")
        if not recommendations:
            recommendations.append("Maintain optimal metabolic health with regular resistance training and circadian alignment.")

    insight = (
        gemini_data.get("actuarial_insight")
        if gemini_data and gemini_data.get("actuarial_insight")
        else f"At {clean_age} years old in {country}, your lifestyle profile projects a lifespan of {projected_lifespan:.1f} years ({target_year}). You have approximately {days_remaining:,} days remaining."
    )

    longevity_score = (
        gemini_data.get("longevity_score", int(max(10, min(99, 50 + delta_years * 5))))
        if gemini_data
        else int(max(10, min(99, 50 + delta_years * 5)))
    )

    return {
        "status": "success",
        "current_age": clean_age,
        "biological_age": round(bio_age, 1),
        "projected_lifespan_years": round(projected_lifespan, 1),
        "years_remaining": round(years_remaining, 1),
        "days_remaining": days_remaining,
        "hours_remaining": hours_remaining,
        "seconds_remaining": seconds_remaining,
        "projected_death_year": target_year,
        "projected_death_date": estimated_target_date.strftime("%B %d, %Y"),
        "country_baseline_lifespan": round(baseline_lifespan, 1),
        "lifestyle_year_delta": round(delta_years, 1),
        "longevity_score": longevity_score,
        "actuarial_insight": insight,
        "top_enhancers": enhancers,
        "top_hazards": hazards,
        "actionable_recommendations": recommendations,
        "engine": "Gemini Cloud API (actuarial)" if gemini_data else "Deterministic Epidemiological Actuarial Model",
    }


def register_death_clock_tools(engine: RobotEngine) -> None:
    """Register system.death_clock tool in the ToolRegistry.

    Args:
        engine: Active RobotEngine instance.
    """
    registry = getattr(engine, "tool_registry", None) or getattr(engine, "tools", None)
    if not registry:
        return

    registry.register(
        ToolDefinition(
            name="system.death_clock",
            description=(
                "Estimate actuarial life expectancy, projected death date, and longevity countdown "
                "based on age, country, and lifestyle factors (exercise, diet, sleep, smoking, stress) "
                "using epidemiological tables and Gemini AI clinical analysis."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "Current age in years (e.g. 20).",
                        "default": 20,
                    },
                    "gender": {
                        "type": "string",
                        "description": "Biological sex or gender ('male', 'female').",
                        "default": "male",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country of residence for actuarial baseline (e.g. 'India', 'USA').",
                        "default": "India",
                    },
                    "smoking": {
                        "type": "string",
                        "description": "Smoking status ('never', 'occasional', 'heavy').",
                        "default": "never",
                    },
                    "exercise_days_per_week": {
                        "type": "integer",
                        "description": "Weekly exercise frequency from 0 to 7.",
                        "default": 3,
                    },
                    "diet_quality": {
                        "type": "string",
                        "description": "Diet nutritional quality ('poor', 'average', 'healthy', 'exceptional').",
                        "default": "average",
                    },
                    "sleep_hours": {
                        "type": "number",
                        "description": "Average hours of sleep per night (e.g. 7.5).",
                        "default": 7.5,
                    },
                    "stress_level": {
                        "type": "string",
                        "description": "Chronic stress rating ('low', 'moderate', 'high', 'extreme').",
                        "default": "moderate",
                    },
                    "alcohol": {
                        "type": "string",
                        "description": "Alcohol intake rating ('none', 'moderate', 'heavy').",
                        "default": "moderate",
                    },
                    "bmi": {
                        "type": "number",
                        "description": "Optional body mass index (e.g. 22.5).",
                    },
                },
            },
            handler=estimate_death_clock,
            permission_level=PermissionLevel.READ_ONLY,
            family="system",
            is_idempotent=True,
            timeout_s=7.0,
        )
    )
    if hasattr(registry, "ALIASES"):
        registry.ALIASES["death_clock"] = "system.death_clock"
        registry.ALIASES["health.death_clock"] = "system.death_clock"
        registry.ALIASES["system_death_clock"] = "system.death_clock"

