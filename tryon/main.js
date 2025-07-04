// let dresses = [];
// let selectedDress = null;
// let selectedColor = null;

// document.addEventListener("DOMContentLoaded", async () => {
//   await loadDressData();
//   renderGallery("all");
// });

// function toggleMenu() {
//   const menu = document.getElementById("menuDropdown");
//   menu.classList.toggle("hidden");
// }

// function toggleSubMenu(id) {
//   const submenu = document.getElementById(id);
//   submenu.classList.toggle("hidden");
// }

// function closeAllMenus() {
//   document.getElementById("menuDropdown").classList.add("hidden");
//   document.getElementById("clothesSubMenu").classList.add("hidden");
//   document.getElementById("accessoriesSubMenu").classList.add("hidden");
// }

// async function loadDressData() {
//   try {
//     const res = await fetch("/static/data/dress.json");
//     if (!res.ok) throw new Error("Failed to load dress data.");
//     dresses = await res.json();
//   } catch (error) {
//     console.error("Error loading dresses:", error);
//     document.getElementById(
//       "app"
//     ).innerHTML = `<p>Error loading dress data: ${error.message}</p>`;
//   }
// }

// function filterByCategory(category) {
//   renderGallery(category);
// }

// function renderGallery(category) {
//   const app = document.getElementById("app");
//   app.innerHTML = "";

//   const gallery = document.createElement("div");
//   gallery.className = "dress-gallery";

//   dresses
//     .filter((d) => category === "all" || d.type === category)
//     .forEach((dress) => {
//       const item = document.createElement("div");
//       item.className = "dress-item";
//       item.innerHTML = `
//           <img src="${dress.baseImage}" alt="${dress.name}" width="200">
//           <p>${dress.name}</p>
//           <button onclick="selectDress(${dress.id})">Try On</button>
//         `;
//       gallery.appendChild(item);
//     });

//   app.appendChild(gallery);
// }

// function selectDress(dressId) {
//   selectedDress = dresses.find((d) => d.id === dressId);
//   const colors = selectedDress.colors
//     .map((color) => {
//       const path = `/static/images/${selectedDress.type}${selectedDress.id}_${color}.png`;
//       return `<img src="${path}" onclick="selectColor('${color}', '${path}')" width="120" style="margin:10px;cursor:pointer;">`;
//     })
//     .join("");

//   document.getElementById("app").innerHTML = `
//       <h2>Select a Color for ${selectedDress.name}</h2>
//       <div class="color-images">${colors}</div>
//       <div id="color-selection-feedback"></div>
//     `;
// }

// function selectColor(color, imagePath) {
//   selectedColor = color;
//   selectedDress.baseImage = imagePath;

//   document.getElementById("app").innerHTML = `
//       <p>Selected Color: <strong>${color}</strong></p>
//   `;

//   // Directly show upload section
//   showUploadSection();
// }

// function showUploadSection() {
//   const app = document.getElementById("app");
//   app.innerHTML = `
//       <h2>Upload Your Full-Length Photo</h2>
//       <input type="file" id="user-photo" accept="image/*" onchange="previewImage(event)">
//       <button onclick="processTryOn()">Generate Try-On</button>
//       <div id="preview-container"></div>
//       <div id="loading" class="hidden">Processing your image...</div>
//     `;
// }

// function previewImage(event) {
//   const file = event.target.files[0];
//   const reader = new FileReader();
//   reader.onload = (e) => {
//     document.getElementById("preview-container").innerHTML = `
//         <img src="${e.target.result}" class="preview-image">
//       `;
//   };
//   if (file) reader.readAsDataURL(file);
// }

// async function processTryOn() {
//   const fileInput = document.getElementById("user-photo");
//   if (fileInput.files.length === 0) {
//     alert("Please upload your photo first");
//     return;
//   }

//   const file = fileInput.files[0];
//   const formData = new FormData();
//   formData.append("file", file);

//   try {
//     const uploadRes = await fetch("/upload", {
//       method: "POST",
//       body: formData,
//     });
//     const uploadData = await uploadRes.json();
//     if (!uploadRes.ok) throw new Error(uploadData.error);

//     const tryOnRes = await fetch("/try-on", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         user_image: uploadData.filepath,
//         dress_image: selectedDress.baseImage.replace(/^\/+/, ""),
//       }),
//     });

//     const tryOnData = await tryOnRes.json();
//     if (!tryOnData.success) throw new Error(tryOnData.error);

//     showTryOnResult(tryOnData.result_image);
//   } catch (error) {
//     document.getElementById(
//       "app"
//     ).innerHTML = `<p style="color:red;">${error.message}</p>`;
//   }
// }

// function showTryOnResult(imageUrl) {
//   document.getElementById("app").innerHTML = `
//       <h2>Your Virtual Try-On Result</h2>
//       <img src="${imageUrl}" class="result-image">
//       <div>
//         <button onclick="saveResult('${imageUrl}')">Save Image</button>
//         <button onclick="location.reload()">Try Another</button>
//       </div>
//     `;
// }

// function saveResult(url) {
//   alert("Image saved (demo). Implement download logic here.");
// }

let dresses = [];
let selectedDress = null;
let selectedColor = null;

document.addEventListener("DOMContentLoaded", async () => {
  await loadDressData();
  renderGallery("all");
});

function toggleMenu() {
  const menu = document.getElementById("menuDropdown");
  menu.classList.toggle("hidden");
}

function toggleSubMenu(id) {
  const submenu = document.getElementById(id);
  submenu.classList.toggle("hidden");
}

function closeAllMenus() {
  document.getElementById("menuDropdown").classList.add("hidden");
  document.getElementById("clothesSubMenu").classList.add("hidden");
  document.getElementById("accessoriesSubMenu").classList.add("hidden");
}

async function loadDressData() {
  try {
    const res = await fetch("/static/data/dress.json");
    if (!res.ok) throw new Error("Failed to load dress data.");
    dresses = await res.json();
  } catch (error) {
    console.error("Error loading dresses:", error);
    document.getElementById(
      "app"
    ).innerHTML = `<p>Error loading dress data: ${error.message}</p>`;
  }
}

function filterByCategory(category) {
  renderGallery(category);
}

function renderGallery(category) {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const gallery = document.createElement("div");
  gallery.className = "dress-gallery";

  dresses
    .filter((d) => category === "all" || d.type === category)
    .forEach((dress) => {
      const item = document.createElement("div");
      item.className = "dress-item";
      item.innerHTML = `
        <img src="${dress.baseImage}" alt="${dress.name}" width="200">
        <p>${dress.name}</p>
        <button onclick="selectDress(${dress.id})">Try On</button>
      `;
      gallery.appendChild(item);
    });

  app.appendChild(gallery);
}

function selectDress(dressId) {
  selectedDress = dresses.find((d) => d.id === dressId);
  const colors = selectedDress.colors
    .map((color) => {
      const path = `/static/images/${selectedDress.type}${selectedDress.id}_${color}.png`;
      return `<img src="${path}" onclick="selectColor('${color}', '${path}')" width="120" style="margin:10px;cursor:pointer;">`;
    })
    .join("");

  document.getElementById("app").innerHTML = `
    <h2>Select a Color for ${selectedDress.name}</h2>
    <div class="color-images">${colors}</div>
  `;
}

function selectColor(color, imagePath) {
  selectedColor = color;
  selectedDress.baseImage = imagePath;

  document.getElementById("app").innerHTML = `
    <p>Selected Color: <strong>${color}</strong></p>
  `;

  showUploadSection();
}

function showUploadSection() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <h2>Upload Your Full-Length Photo</h2>
    <input type="file" id="user-photo" accept="image/*" onchange="previewImage(event)">
    <button onclick="processTryOn()">Generate Try-On</button>
    <div id="preview-container"></div>
    <div id="loading" class="hidden">Processing your image...</div>
  `;
}

function previewImage(event) {
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("preview-container").innerHTML = `
      <img src="${e.target.result}" class="preview-image">
    `;
  };
  if (file) reader.readAsDataURL(file);
}

async function processTryOn() {
  const fileInput = document.getElementById("user-photo");
  if (fileInput.files.length === 0) {
    alert("Please upload your photo first");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  try {
    // Upload image to server
    const uploadRes = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const uploadData = await uploadRes.json();
    if (!uploadRes.ok) throw new Error(uploadData.error);

    // Send both user image path and dress image path to /try-on
    const tryOnRes = await fetch("/try-on", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_image: uploadData.filepath,
        dress_image: selectedDress.baseImage.replace(/^\/+/, ""),
        mode: selectedDress.mode,
      }),
    });

    const tryOnData = await tryOnRes.json();
    if (!tryOnData.success) throw new Error(tryOnData.error);

    showTryOnResult(tryOnData.result_image);
  } catch (error) {
    document.getElementById(
      "app"
    ).innerHTML = `<p style="color:red;">${error.message}</p>`;
  }
}

function showTryOnResult(imageUrl) {
  document.getElementById("app").innerHTML = `
    <h2>Your Virtual Try-On Result</h2>
    <img src="${imageUrl}" class="result-image">
    <div>
      <button onclick="saveResult('${imageUrl}')">Save Image</button>
      <button onclick="location.reload()">Try Another</button>
    </div>
  `;
}

function saveResult(url) {
  alert("Image saved (demo). Implement download logic here.");
}
