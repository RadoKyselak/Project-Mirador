document.addEventListener("mouseup", async (e) => {
  const selection = window.getSelection().toString().trim()
  if (!selection) return

  const existing = document.getElementById("stelthar_verify_prompt")
  if (existing) existing.remove()

  const prompt = document.createElement("div")
  prompt.id = "stelthar_verify_prompt"

  // Prompt Style
  Object.assign(prompt.style, {
    position: "absolute",
    zIndex: "2147483647",
    background: "#1a1a1a",
    color: "#f0eadd",
    padding: "8px 12px",
    borderRadius: "6px",
    fontSize: "12px",
    fontWeight: "bold",
    cursor: "pointer",
    fontFamily: "'Courier New', Courier, monospace",
    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
    transition: "transform 0.2s ease, box-shadow 0.2s ease",
    letterSpacing: "0.5px",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    border: "1px solid #444"
  })
  prompt.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke="#f0eadd"
         stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M9 11l3 3L22 4"></path>
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
    </svg>
    <span>Verify with Stelthar</span>
  `
  document.body.appendChild(prompt)
  const rect = window.getSelection().getRangeAt(0).getBoundingClientRect()
  prompt.style.top = window.scrollY + rect.top - 45 + "px"
  prompt.style.left = window.scrollX + rect.left + "px"

  // Hover Effect
  prompt.addEventListener("mouseenter", () => {
    prompt.style.transform = "translateY(-1px)";
    prompt.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.4)";
  })
  prompt.addEventListener("mouseleave", () => {
    prompt.style.transform = "translateY(0)";
    prompt.style.boxShadow = "0 2px 8px rgba(0, 0, 0, 0.3)";
  })
  const cleanup = () => {
    const el = document.getElementById("stelthar_verify_prompt")
    if (el) el.remove()
  }
  prompt.onclick = () => {
    cleanup()
    chrome.runtime.sendMessage({ type: "verify", claim: selection })
  }
  setTimeout(cleanup, 5000)
})
