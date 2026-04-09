# Reference channel configs.
# video_urls: manually curated list of top-performing video URLs per channel.
# Add URLs before running build_kb.py. Target ~20 per channel.

REFERENCE_CHANNELS: dict = {
    "johnny_harris": {
        "name": "Johnny Harris",
        "focus": "storytelling, emotional arc, host-driven documentary style",
        "video_urls": [],  # populate before running KB build
        "target_count": 20,
    },
    "wendover": {
        "name": "Wendover Productions",
        "focus": "research depth, information pacing, visual evidence",
        "video_urls": [],
        "target_count": 20,
    },
    "kurzgesagt": {
        "name": "Kurzgesagt",
        "focus": "hook mastery, script structure, retention engineering",
        "video_urls": [],
        "target_count": 20,
    },
    "veritasium": {
        "name": "Veritasium",
        "focus": "curiosity-gap hooks, surprise reveals, viewer psychology",
        "video_urls": [],
        "target_count": 20,
    },
    "reallifelore": {
        "name": "RealLifeLore",
        "focus": "geographic/analytical framing, b-roll discipline",
        "video_urls": [],
        "target_count": 20,
    },
}

CHANNEL_PROFILES: dict = {
    "predictive_echoes": {
        "domain": "geopolitics, technology, future trends",
        "tone": "analytical, slightly urgent, forward-looking",
        "arc_template": "trend_identification → implication → future_scenario",
        "visual_style": "data-heavy, news footage, maps, graphs",
        "north_star": "make the viewer see something coming they hadn't noticed",
        "reference_channels": ["wendover", "reallifelore"],
        "avg_length_min": 10,
    },
    "incentives_lab": {
        "domain": "economics, human behavior, incentive structures",
        "tone": "curious, slightly provocative, Munger-esque",
        "arc_template": "surface_reality → hidden_incentive → reframe",
        "visual_style": "human-focused, institutional, everyday scenes",
        "north_star": "make the viewer distrust something they trusted before",
        "reference_channels": ["johnny_harris", "kurzgesagt"],
        "avg_length_min": 12,
    },
}
