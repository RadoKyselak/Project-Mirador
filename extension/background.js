// background.js
const BACKEND_URL = "https://your-project.vercel.app/verify"; // Replace with your actual Vercel deployment URL

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "verify") {
    // open the popup (action) and store the claim in storage for popup to read
    chrome.storage.local.set({ "stelthar_last_claim": message.claim }, () => {
      // optionally open the popup UI
      chrome.action.openPopup(() => {
        // popup will read the stored claim and call backend
      });
    });
  }
  return true;
});
