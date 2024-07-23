const toolbar = document.getElementById('toolbar');
let icons = {
    "bench": "bench-svgrepo-com.svg",
    "bicycle" : "bicycle-svgrepo-com.svg",
    "boat": "boat-with-a-sail-svgrepo-com.svg",
    "book" : "book-2-svgrepo-com.svg",
    "cake" : "cake-svgrepo-com.svg",
    "car" : "car-svgrepo-com.svg",
    "chair" : "chair-free-2-svgrepo-com.svg",
    "child" : "child-svgrepo-com.svg",
    "coffe" : "coffe-cup-svgrepo-com.svg",
    "house" : "house-svgrepo-com.svg",
    "human"  : "human-boy-person-man-svgrepo-com.svg",
    "laptop" : "laptop-alt-1-svgrepo-com.svg",
    "man" : "man-svgrepo-com.svg",
    "motorcycle": "motorcycle-cross-moto-bike-svgrepo-com.svg"
}

for (let icon in icons) {
    let btn = document.createElement('button');
    btn.width = 20;
    btn.height = 20;
    btn.classList.add('draggable');
    btn.setAttribute('data-name', icon);
    var img = document.createElement('img');
    img.src = `src/${icons[icon]}`;
    img.width = 20;
    img.height = 20;
    img.alt = icon;
    btn.appendChild(img);
    toolbar.appendChild(btn);
}

const whiteboard = document.getElementById('whiteboard');
const draggables = document.querySelectorAll('.draggable');
const clearBtn = document.querySelector(".buttonclr");
const submitBtn = document.querySelector(".buttonsubmit");

draggables.forEach(draggable => {
    draggable.addEventListener('dragstart', dragStart);
    draggable.addEventListener('dragend', dragEnd);
});

whiteboard.addEventListener('dragover', dragOver);
whiteboard.addEventListener('drop', drop);

clearBtn.addEventListener('click', clear)
submitBtn.addEventListener('click', submit)

let draggedItem = null;

function dragStart(e) {
    draggedItem = e.target;
    setTimeout(() => {
        e.target.style.display = 'block';
    }, 0);
}

function dragStartinWB(e) {
    draggedItem = e.target;
    setTimeout(() => {
        e.target.style.display = 'none';
    }, 0);
}



function dragEnd(e) {
    setTimeout(() => {
        e.target.style.display = 'block';
        draggedItem = null;
    }, 0);
}

function dragOver(e) {
    e.preventDefault();
}

function drop(e) {
    e.preventDefault();
    const clone = draggedItem.cloneNode(true);
    clone.style.position = 'absolute';
    clone.style.left = `${e.clientX - whiteboard.offsetLeft - clone.offsetWidth / 2}px`;
    clone.style.top = `${e.clientY - whiteboard.offsetTop - clone.offsetHeight / 2}px`;
    // clone.classList.add(`${clone.alt}`)
    whiteboard.appendChild(clone);
    makeDraggable(clone);
}

function makeDraggable(element) {
    element.addEventListener('dragstart', dragStartinWB);
    element.addEventListener('dragend', dragEnd);
}

function clear() {
    whiteboard.innerHTML = '';
}

function submit() {
    var children = whiteboard.children;
    console.log(children);
    for (let i in children) {
        console.log(children[i])
    }
}
