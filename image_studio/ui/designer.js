
(payload) => {
  let data = {};
  try {
    data = JSON.parse(payload || "{}");
  } catch (error) {
    console.error("Could not parse Ideogram JSON designer payload", error);
  }

  const setNativeValue = (element, value) => {
    const proto = element instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const descriptor = Object.getOwnPropertyDescriptor(proto, "value");
    if (descriptor && descriptor.set) descriptor.set.call(element, value);
    else element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const setTextControl = (elemId, value) => {
    const root = document.getElementById(elemId);
    if (!root) return false;
    const control = root.querySelector("textarea, input");
    if (!control) return false;
    setNativeValue(control, value);
    return true;
  };

  const clickChoice = (elemId, labelText) => {
    const root = document.getElementById(elemId);
    if (!root) return false;
    const normalized = String(labelText || "").trim();
    for (const candidate of root.querySelectorAll("label, button")) {
      if (String(candidate.textContent || "").trim() === normalized) {
        candidate.click();
        return true;
      }
    }
    for (const input of root.querySelectorAll("input")) {
      if (String(input.value || "").trim() === normalized) {
        if (!input.checked) input.click();
        return true;
      }
    }
    return false;
  };

  const setCheckbox = (elemId, checked) => {
    const root = document.getElementById(elemId);
    const input = root && root.querySelector("input[type='checkbox']");
    if (!input) return false;
    if (Boolean(input.checked) !== Boolean(checked)) input.click();
    return true;
  };

  if (!window.__ideogramJsonDesignerBridgeInstalled) {
    window.addEventListener("message", (event) => {
      if (event.origin !== window.location.origin) return;
      const message = event.data || {};
      if (message.type === "ideogram4-json-designer-save" && message.caption) {
        setTextControl("gen-prompt", message.caption);
        clickChoice("gen-ideogram-upsampler", "None");
        setCheckbox("gen-ideogram-strip-prompt", false);
      }
    });
    window.__ideogramJsonDesignerBridgeInstalled = true;
  }

  const url = data.url || "/ideogram4/json-designer";
  const editor = window.open(url, "ideogram4_json_designer", "popup,width=1440,height=980");
  if (!editor) {
    console.warn("The Ideogram JSON designer popup was blocked.");
    return;
  }

  const sendPayload = () => {
    editor.postMessage(
      { type: "ideogram4-json-designer-load", payload: data },
      window.location.origin,
    );
  };

  const onReady = (event) => {
    if (event.origin !== window.location.origin || event.source !== editor) return;
    if (!event.data || event.data.type !== "ideogram4-json-designer-ready") return;
    sendPayload();
    window.removeEventListener("message", onReady);
  };
  window.addEventListener("message", onReady);
  setTimeout(sendPayload, 500);
}
