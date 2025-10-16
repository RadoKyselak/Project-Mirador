const BACKEND_URL = "https://stelthar-api.vercel.app";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "verify") {
    chrome.storage.local.set({ "stelthar_last_claim": message.claim }, () => {
      chrome.action.openPopup(() => {
      });
    });
  }
  return true;
});



