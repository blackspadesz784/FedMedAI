/**
 * app.js
 * App-shell behaviour shared by every view: auth guard, sidebar navigation,
 * sidebar collapse toggle, profile chip, and the toast notification helper.
 */

// ---- Auth guard: initialize default session if none present so opening index.html works immediately ----
let doctor = JSON.parse(sessionStorage.getItem("fedrad_doctor") || "null");
if (!doctor) {
  doctor = { name: "Dr. Aanya Mehta", email: "dr.mehta@stmarcus-hosp.org" };
  sessionStorage.setItem("fedrad_doctor", JSON.stringify(doctor));
}

document.addEventListener("DOMContentLoaded", () => {
  mountIcons();

  if (doctor) {
    document.getElementById("profileName").textContent = doctor.name || "Doctor";
    const initials = (doctor.name || "D R")
      .split(" ")
      .map((s) => s[0])
      .filter(Boolean)
      .slice(0, 2)
      .join("")
      .toUpperCase();
    document.getElementById("profileInitials").textContent = initials || "DR";
  }

  // ---- Sidebar navigation / view routing ----
  const navItems = document.querySelectorAll(".nav-item[data-view]");
  const views = document.querySelectorAll(".view");
  const titleMap = {
    dashboard: ["Overview", `Good to see you, ${doctor?.name?.split(" ")[0] || "Doctor"}`],
    upload: ["Clinical", "Upload & predict"],
    history: ["Clinical", "Patient history"],
    "disease-stats": ["Analytics", "Disease statistics"],
    "dataset-stats": ["Analytics", "Dataset statistics"],
    visualizations: ["Analytics", "Data visualizations"],
    "fl-monitor": ["Federated learning", "FL monitor"],
    "model-performance": ["Federated learning", "Model performance"],
    about: ["Project", "About FedRad"],
  };

  function goToView(name) {
    navItems.forEach((n) => n.classList.toggle("active", n.dataset.view === name));
    views.forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
    const [eyebrow, title] = titleMap[name] || ["", ""];
    document.getElementById("topbarEyebrow").textContent = eyebrow;
    document.getElementById("topbarTitle").textContent = title;
    window.location.hash = name;
    document.dispatchEvent(new CustomEvent("view:shown", { detail: { name } }));
  }

  navItems.forEach((item) => item.addEventListener("click", () => goToView(item.dataset.view)));

  const initial = (window.location.hash || "#dashboard").replace("#", "");
  goToView(titleMap[initial] ? initial : "dashboard");

  // ---- Sidebar collapse ----
  document.getElementById("collapseBtn").addEventListener("click", () => {
    document.getElementById("appShell").classList.toggle("collapsed");
  });

  // ---- Sign out ----
  document.getElementById("logoutBtn").addEventListener("click", () => {
    sessionStorage.removeItem("fedrad_doctor");
    window.location.href = "login.html";
  });
});

// ---- Toast helper (used across views for success/error feedback) ----
function showToast(message, type = "success") {
  const stack = document.getElementById("toastStack");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span data-icon="${type === "success" ? "check" : "alert"}"></span><span>${message}</span>`;
  stack.appendChild(el);
  mountIcons(el);
  setTimeout(() => {
    el.style.transition = "opacity .3s, transform .3s";
    el.style.opacity = "0";
    el.style.transform = "translateX(20px)";
    setTimeout(() => el.remove(), 300);
  }, 3500);
}
