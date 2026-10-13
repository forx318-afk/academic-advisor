import pytest
from app import app

GOOD = dict(study_hours=25, attendance=95, prev_gpa=4.5, assignments=95,
            sleep_hours=7.5, extra_hours=4, failed_courses=0)
BAD = dict(study_hours=2, attendance=40, prev_gpa=1.5, assignments=30,
           sleep_hours=4, extra_hours=18, failed_courses=4)


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    assert client.get("/health").json["status"] == "ok"


def test_home_page(client):
    assert client.get("/").status_code == 200


def test_strong_student(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200 and r.json["standing"] in ("Excellent", "Good Standing")


def test_weak_student_gets_advice(client):
    r = client.post("/predict", json=BAD)
    assert r.json["standing"] == "At Risk" and len(r.json["advice"]) >= 3


def test_missing_field(client):
    assert client.post("/predict", json={"study_hours": 5}).status_code == 400
