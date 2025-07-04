document.addEventListener("DOMContentLoaded", function () {
  const imageUpload = document.getElementById("imageUpload");
  const previewImg = document.getElementById("uploadedImage");
  const uploadForm = document.getElementById("uploadForm");
  const userPhoto = document.getElementById("user-photo");
  const resultSection = document.getElementById("result");
  const resultImg = document.getElementById("result-img");
  const applyBtn = document.getElementById("applyBtn");
  const accessoryRowContainer = document.getElementById("accessory-rows");

  let uploadedImagePath = "";

  // ✅ SPA navigation
  window.showSection = function (sectionId) {
    document.querySelectorAll(".spa-section").forEach((sec) =>
      sec.classList.remove("active-section")
    );
    document.getElementById(sectionId).classList.add("active-section");
  };

  // ✅ Image preview
  window.previewImage = function (event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        previewImg.src = e.target.result;
        previewImg.style.display = "block";
      };
      reader.readAsDataURL(file);
    }
  };

  // ✅ Upload & get recommendations
  uploadForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const file = imageUpload.files[0] || document.getElementById("imageupload").files[0];
    if (!file) {
      alert("Please upload an image.");
      return;
    }

    resultSection.style.display = "none"; // hide old result

    // Upload image
    const formData = new FormData();
    formData.append("file", file);

    const uploadRes = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const uploadData = await uploadRes.json();
    uploadedImagePath = uploadData.user_image;

    // Get accessory recommendations
    const recommendRes = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_image: uploadedImagePath }),
    });
    const result = await recommendRes.json();
    const accessories = result.accessories;

    userPhoto.src = "/" + uploadedImagePath.replace(/\\/g, "/");
    accessoryRowContainer.innerHTML = "";

    const top_k = 4; // number of sets
    for (let i = 0; i < top_k; i++) {
      const rowDiv = document.createElement("div");
      rowDiv.className = "accessory-row";

      // Title
      const title = document.createElement("h4");
      title.textContent = `Set ${i + 1}`;
      rowDiv.appendChild(title);

      for (const [type, paths] of Object.entries(accessories)) {
        const itemDiv = document.createElement("div");
        itemDiv.className = "accessory-preview";

        const imgPath = paths[i] || paths[0];
        

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "accessory-checkbox";
        checkbox.value = imgPath;
        checkbox.dataset.type = type;
        checkbox.id = `set${i}-${type}`;

        const label = document.createElement("label");
        label.setAttribute("for", `set${i}-${type}`);

        const nameSpan = document.createElement("span");
        nameSpan.className = "accessory-name";
        nameSpan.textContent = type;

        const img = document.createElement("img");
        //const path = paths[i] || paths[0]; // fallback
        img.src = "/" + imgPath.replace(/\\/g, "/");
        img.alt = type;


        label.appendChild(nameSpan);
        label.appendChild(img);
        //label.appendChild(img);

        itemDiv.appendChild(checkbox);
        itemDiv.appendChild(label);
        rowDiv.appendChild(itemDiv);
      }

      accessoryRowContainer.appendChild(rowDiv);
    }

    showSection("recommendations");
  });

  // ✅ Apply Accessories
  applyBtn.addEventListener("click", async function () {
    const checkedAccessories = document.querySelectorAll(".accessory-checkbox:checked");
    if (checkedAccessories.length === 0) {
      alert("Please select at least one accessory.");
      return;
    }

    const selectedAccessories = {};
    Array.from(checkedAccessories).forEach(cb => {
      selectedAccessories[cb.dataset.type] = cb.value;
    });

    const applyRes = await fetch("/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_image : uploadedImagePath,
        selected_accessories: selectedAccessories })
    });

    const applyData = await applyRes.json();
    if (applyData.result_image) {
      resultImg.src = applyData.result_image;
      resultSection.style.display = "block";
    }
  });

});

// ✅ Reapply Accessories
window.reapplyAccessories = async function () {
  const applyRes = await fetch("/apply", { method: "POST" });
  const applyData = await applyRes.json();

  if (applyData.result_image) {
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = `
      <h2>✨ Final Result with Accessories Applied</h2>
      <img src="${applyData.result_image}" alt="Result Image" />
    `;
    resultDiv.style.display = "block";
  }
};
