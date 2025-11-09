async function refreshStatus() {
  const res = await fetch('/status');
  const data = await res.json();
  const table = document.getElementById('userTable');
  data.users.forEach(u => {
    const row = table.querySelector(`tr[data-id='${u.id}']`);
    if (row) {
      row.children[2].innerText = u.is_active ? "🟢 Running" : "🔴 Stopped";
    }
  });
}

async function viewLogs(uid) {
  const res = await fetch(`/logs/${uid}`);
  const txt = await res.text();
  document.getElementById('logBox').innerText = txt;
}

setInterval(refreshStatus, 5000);
