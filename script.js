(function () {
  const epochs = [
    { epoch: 1, loss: 0.516, f1: 0.793, acc: 0.795 },
    { epoch: 2, loss: 0.532, f1: 0.824, acc: 0.830 },
    { epoch: 3, loss: 0.604, f1: 0.844, acc: 0.848 },
    { epoch: 4, loss: 0.708, f1: 0.842, acc: 0.846 }
  ];

  const labels = ["Negative", "Neutral", "Positive"];
  const matrix = [
    [158, 29, 10],
    [15, 316, 29],
    [8, 32, 203]
  ];

  const toggle = document.querySelector(".nav-toggle");
  const nav = document.querySelector(".site-nav");

  if (toggle && nav) {
    toggle.addEventListener("click", function () {
      const open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    });

    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
        toggle.setAttribute("aria-label", "Open menu");
      });
    });
  }

  drawEpochChart(document.getElementById("epoch-chart"), epochs);
  renderMatrix(document.getElementById("confusion"), matrix, labels);

  function drawEpochChart(canvas, data) {
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 720;
    const cssHeight = 360;
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = { top: 18, right: 16, bottom: 42, left: 44 };
    const w = cssWidth - pad.left - pad.right;
    const h = cssHeight - pad.top - pad.bottom;
    const minY = 0.76;
    const maxY = 0.86;

    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = "#b7aa98";
    ctx.font = "12px Segoe UI, system-ui, sans-serif";

    for (let i = 0; i < 6; i += 1) {
      const yVal = minY + ((maxY - minY) * i) / 5;
      const y = pad.top + h - ((yVal - minY) / (maxY - minY)) * h;
      ctx.strokeStyle = "#3a3228";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + w, y);
      ctx.stroke();
      ctx.fillText(yVal.toFixed(2), 8, y + 4);
    }

    const xAt = function (i) {
      return pad.left + (w * i) / (data.length - 1);
    };
    const yAt = function (v) {
      return pad.top + h - ((v - minY) / (maxY - minY)) * h;
    };

    function polyline(key, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      data.forEach(function (row, i) {
        const x = xAt(i);
        const y = yAt(row[key]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = color;
      data.forEach(function (row, i) {
        ctx.beginPath();
        ctx.arc(xAt(i), yAt(row[key]), i === 2 ? 5.5 : 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    polyline("f1", "#e8a04a");
    polyline("acc", "#3dba8b");

    data.forEach(function (row, i) {
      ctx.fillStyle = "#b7aa98";
      ctx.textAlign = "center";
      ctx.fillText("E" + row.epoch, xAt(i), cssHeight - 22);
    });

    ctx.textAlign = "left";
    ctx.fillStyle = "#e8a04a";
    ctx.fillRect(pad.left, cssHeight - 14, 10, 10);
    ctx.fillStyle = "#f4efe6";
    ctx.fillText("Macro F1", pad.left + 16, cssHeight - 5);
    ctx.fillStyle = "#3dba8b";
    ctx.fillRect(pad.left + 100, cssHeight - 14, 10, 10);
    ctx.fillStyle = "#f4efe6";
    ctx.fillText("Accuracy", pad.left + 116, cssHeight - 5);
  }

  function renderMatrix(root, values, classNames) {
    if (!root) return;
    root.innerHTML = "";

    const corner = document.createElement("div");
    corner.className = "matrix-label";
    corner.textContent = "true \\ pred";
    root.appendChild(corner);

    classNames.forEach(function (name) {
      const el = document.createElement("div");
      el.className = "matrix-label";
      el.textContent = name;
      root.appendChild(el);
    });

    const max = Math.max.apply(null, values.flat());
    values.forEach(function (row, r) {
      const lab = document.createElement("div");
      lab.className = "matrix-label";
      lab.textContent = classNames[r];
      root.appendChild(lab);
      row.forEach(function (n) {
        const cell = document.createElement("div");
        cell.className = "matrix-cell";
        cell.textContent = String(n);
        const t = n / max;
        const light = 88 - t * 48;
        cell.style.background = "hsl(32, 72%, " + light + "%)";
        root.appendChild(cell);
      });
    });
  }

  window.addEventListener("resize", function () {
    drawEpochChart(document.getElementById("epoch-chart"), epochs);
  });
})();
