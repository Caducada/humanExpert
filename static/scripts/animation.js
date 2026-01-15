document.querySelector('.dripToggle').addEventListener('click', function() {
    const element = document.querySelector('.dripDrop');
    if (element.style.height === '0px' || !element.style.height) {
        element.style.height = '255px';
        element.style.width = "70%";
        element.style.borderRadius = "25px";
    } else {
        element.style.height = '0px';
    }
    void element.offsetHeight;
});