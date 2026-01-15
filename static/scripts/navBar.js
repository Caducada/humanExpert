const toggle = document.querySelector(".dripToggle");
const menu = document.querySelector(".dripDrop");

toggle.addEventListener("click", () => {
    menu.classList.toggle("open");
});