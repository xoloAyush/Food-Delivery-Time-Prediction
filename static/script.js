const form = document.getElementById("form");
const submitBtn = document.getElementById("submit");
const minutesEl = document.getElementById("minutes");
const messageEl = document.getElementById("message");
const summaryEl = document.getElementById("summary");
const fillEl = document.getElementById("fill");
const riderEl = document.getElementById("rider");
const etaEl = document.querySelector(".eta");

const NUMBER_FIELDS = [
  "Delivery_person_Age", "Delivery_person_Ratings", "distance_km",
  "order_hour", "Vehicle_condition", "Road_traffic_density",
  "multiple_deliveries",
];

function readForm() {
  const data = {};
  new FormData(form).forEach((value, key) => {
    data[key] = NUMBER_FIELDS.includes(key) ? Number(value) : value;
  });
  return data;
}

function describe(minutes) {
  if (minutes <= 20) return "Quick one. Expect it fast.";
  if (minutes <= 35) return "A normal delivery time.";
  if (minutes <= 50) return "A bit slow. Traffic or distance is adding time.";
  return "A long wait. Conditions are working against this order.";
}

function showError(text) {
  messageEl.textContent = text;
  messageEl.classList.add("error");
}

function animateRoute(minutes) {
  // 10 min = almost there, 70 min = far away
  const progress = Math.min(Math.max((70 - minutes) / 60, 0.08), 0.95);
  const pct = (progress * 100).toFixed(0) + "%";
  fillEl.style.width = pct;
  riderEl.style.left = pct;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = readForm();
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";
  messageEl.classList.remove("error");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error("The server rejected this input (status " + res.status + ").");
    }

    const result = await res.json();
    const minutes = Math.round(result.predicted_delivery_time_min);

    minutesEl.textContent = minutes;
    etaEl.classList.remove("pop");
    void etaEl.offsetWidth; // restart the animation
    etaEl.classList.add("pop");

    messageEl.textContent = describe(minutes);
    summaryEl.textContent =
      payload.distance_km + " km by " + payload.Type_of_vehicle.replace("_", " ") +
      ", " + payload.Weatherconditions.toLowerCase() + " weather in a " +
      payload.City.toLowerCase() + " area.";
    animateRoute(minutes);
  } catch (err) {
    showError(err.message || "Could not reach the server. Check that it is running and try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict delivery time";
  }
});