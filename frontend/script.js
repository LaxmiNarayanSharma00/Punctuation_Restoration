document.getElementById("restoreBtn").addEventListener("click", async () => {
  const inputText = document.getElementById("inputText").value;

  const response = await fetch("http://127.0.0.1:8000/restore", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: inputText })
  });

  const data = await response.json();
  document.getElementById("output").innerText = data.restored;
});
