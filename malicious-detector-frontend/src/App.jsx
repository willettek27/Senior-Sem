import { useState } from "react";

function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [logoHover, setLogoHover] = useState(false);

  const handleScan = async () => {
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      setResult("Error scanning URL");
    }
    setLoading(false);
  };

  return (
    <>
      <header
        style={{
          width: "100%",
          background: "#bc667fff",
          boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
          height: "64px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: "100%",
            maxWidth: "1100px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 32px",
          }}
        >
          {/*logo code here*/}
          <div
            style={{
              width: "175px",
              height: "55px",
              background: logoHover ? "#ffffffff" : "#bc667fff",
              border: "2px solid #fff",
              borderRadius: "5px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "background 0.2s, box-shadow 0.2s",
              boxShadow: logoHover ? "0 2px 8px rgba(0,0,0,0.12)" : "none",
              marginRight: "16px",
              cursor: "pointer",
            }}
            onMouseEnter={() => setLogoHover(true)}
            onMouseLeave={() => setLogoHover(false)}
          >
            <span
              style={{
                fontWeight: "bold",
                fontSize: logoHover ? "1.5rem" : "2.5rem",
                color: logoHover ? "#bc667fff" : "#fff",
                letterSpacing: "3px",
                fontFamily: "Segoe UI, Arial, sans-serif",
                transition: "color 0.2s, transform 0.2s",
                transform: logoHover ? "scale(1.15)" : "scale(1)",
                userSelect: "none",
              }}
            >
              Scanify
            </span>
          </div>
          {/* Right-side header content (empty for now) */}
          <div></div>
        </div>
      </header>

      <div
        style={{
          height: "100vh",
          width: "100vw",
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#f0ede9ff",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            textAlign: "center",
            padding: "32px",
            borderRadius: "12px",
            background: "#fff",
            boxShadow: "0 2px 12px rgba(0,0,0,0.07)",
            marginTop: "-60px",
            marginBottom: "30px",
            boxSizing: "border-box",
            width: "100%",
            maxWidth: "1100px",
            height: "380px",
            justifyContent: "flex-start",
          }}
        >
          <h1
            style={{
              marginBottom: "20px",
              fontSize: "40px",
              color: "#bc667fff",
              width: "100%",
              whiteSpace: "normal",
              textOverflow: "ellipsis",
              fontFamily: "Helvetica, Arial, Tahoma, sans-serif",
            }}
          >
            Free website to detect malicious websites
          </h1>
            
          <div style={{ marginBottom: "16px", color: "#555", fontSize: "16px" }}>
            Paste a website URL below and click <b>Scan</b> to check its safety.
          </div>
          <input
            type="text"
            placeholder="Enter URL (e.g., https://example.com)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            style={{
              width: "100%",
              maxWidth: "600px",
              padding: "10px",
              fontSize: "18px",
              marginBottom: "20px",
            }}
            disabled={loading}
          />

          <br />

          <button
            onClick={handleScan}
            style={{
              padding: "10px 35px",
              fontSize: "20px",
              fontFamily: "Arial, sans-serif",
              fontWeight: "bold",
              backgroundColor: "#bc667fff",
              color: "#fff",
              border: "none",
              borderRadius: "2px",
              cursor: "pointer",
            }}
            disabled={loading || !url}
          >
            {loading ? "Scanning..." : "Scan"}
          </button>

          {result && (
            <p
              style={{
                marginTop: "25px",
                fontSize: "20px",
                fontWeight: "bold",
                color:
                  result === "safe"
                    ? "green"
                    : result === "Malicious"
                    ? "red"
                    : "orange",
              }}
            >
              Result: {result}
            </p>
          )}
          {/* Info about the website below the whole box */}
        <div style={{ marginTop: "8px", color: "#888", fontSize: "15px", maxWidth: "600px", textAlign: "center" }}>
          <span role="img" aria-label="info">ℹ️</span> <b>Scanify</b> uses advanced AI to help keep you safe online.<br />
          No URLs are stored. For educational use only.
        </div>
        </div>
      </div>
    </>
  );
}

export default App;