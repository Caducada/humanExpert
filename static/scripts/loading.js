window.addEventListener("load", () => {
  const loader = document.getElementById("loader");
  const container = document.getElementById("container");

  if (loader) {
    loader.classList.add("active");

    setTimeout(() => {
      loader.classList.remove("active");

      setTimeout(() => {
        loader.style.display = "none";
        container.style.display = "block";
        container.classList.add("visible");
      }, 800);
    }, 1600);
  } else {
    container.style.display = "block";
    container.classList.add("visible");
  }
});