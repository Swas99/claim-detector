"""Locust load test for the claim detection API.

Usage:
    # Headless (60s, 50 users):
    make load-test

    # With web UI:
    locust -f tests/locustfile.py --host http://localhost:8000
"""

import random

from locust import HttpUser, between, task


CLAIM_SENTENCES = [
    "The Empire State Building is the tallest building in New York City.",
    "GDP grew 2.1% last quarter according to official statistics.",
    "The unemployment rate fell to 3.5% in December 2023.",
    "The earth orbits the sun at approximately 67,000 miles per hour.",
    "Python is the most popular programming language in 2024.",
    "Amazon reported revenue of $574 billion in fiscal year 2023.",
    "The Great Wall of China is over 13,000 miles long.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
]

NON_CLAIM_SENTENCES = [
    "I think pizza is the best food ever.",
    "What time does the meeting start?",
    "Hello, how are you doing today?",
    "Please close the door when you leave.",
    "I believe we should try a different approach.",
    "Stop making so much noise!",
    "In my opinion, this movie was terrible.",
    "Are you sure about that?",
]

ALL_SENTENCES = CLAIM_SENTENCES + NON_CLAIM_SENTENCES


class ClaimDetectorUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict_single(self):
        self.client.post("/predict", json={
            "text": random.choice(ALL_SENTENCES),
        })

    @task(2)
    def predict_batch(self):
        batch = random.sample(ALL_SENTENCES, k=random.randint(2, 5))
        self.client.post("/predict/batch", json={"texts": batch})

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def model_info(self):
        self.client.get("/model/info")
