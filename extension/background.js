const BACKEND_URL = "https://stelthar-api.vercel.app/verify";
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("Background received message:", message);ng
  if (message.type === "verify" && message.claim) {
    chrome.storage.local.set({ "stelthar_last_claim": message.claim }, () => {
      console.log("Claim saved to storage:", message.claim);
      chrome.action.openPopup(() => {
        if (chrome.runtime.lastError) {
          console.error("Error opening popup:", chrome.runtime.lastError.message);
        } else {
          console.log("Popup opened successfully.");
        }
      });
    });
    return true;
  } else if (message.type === "verify" && !message.claim) {
     console.warn("Received verify message but no claim text.");
  }
});
