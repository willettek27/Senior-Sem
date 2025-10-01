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
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();
      setResult(data.prediction);
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
          height: "80px",
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
      
    {/* Main content  */}
      <div
  style={{
    width: "100%",
    minHeight: "100vh",
    background: "#f0ede9ff",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    paddingTop: "90px",
    paddingBottom: "40px",
    
    }}
    >
    <div
    style={{
      width: "100%",
      maxWidth: "1100px",
      background: "#fff",
      borderRadius: "12px",
      boxShadow: "0 2px 12px rgba(0,0,0,0.07)",
      padding: "32px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      textAlign: "center",
      margin: "0 auto",
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
            
          <div style={{ marginBottom: "30px", marginTop: "10px", color: "#181616ff", fontSize: "23px" }}>
            Paste a website URL below and click <b>Scan</b> to check its safety.
          </div>
          <input
            type="text"
            placeholder="Enter URL (e.g., https://example.com)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            style={{
              width: "100%",
              maxWidth: "650px",
              padding: "10px",
              fontSize: "20px",
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
              marginTop: "-5px",
              borderRadius: "2px",
              cursor: "pointer",
            }}
            disabled={loading || !url}
          >
            {loading ? "Scanning..." : "Scan"}
      </button>

      {result && (
        <p style={{
          marginTop: "25px",
          fontSize: "20px",
          fontWeight: "bold",
          color:
            result === "Benign"
              ? "green"
              : result === "Defacement"
              ? "#ff9800"
              : result === "Phishing"
              ? "#e53935"
              : result === "Malware"
              ? "#6a1b9a"
              : "#fbc626ff",
        }}>
          Result: {result}
        </p>
      )}

    </div>
    <div style={{
      margin: "28px auto 0 auto",
      color: "#706d6dff",
      fontSize: "15px",
      maxWidth: "600px",
      textAlign: "center"
    }}>
      <span role="img" aria-label="info">ℹ️</span> <b>Scanify</b> uses advanced AI to help keep you safe online.<br />
      <span role="img" aria-label="lock">🔒</span> We do not store any URLs or personal data. Your privacy is our priority.
    </div>
  </div>
<div
  style={{
    width: "100vw",
    background: "#f0ede9ff",
    display: "flex",
    justifyContent: "center",
    marginTop: "-200px",
  }}
  >
    <h2
      style={{
        color: "#bc667fff",
        fontSize: "40px",
        fontWeight: "bold",
        textAlign: "center",
        marginBottom: "15px",
        letterSpacing: "1px",
        marginTop: "-65px",
        fontFamily: "Helvetica, Arial, Tahoma, sans-serif"
      }}
      >
        Website Safety Tips & Info

      </h2>
  </div>
<div
  style={{
    display: "flex",
    justifyContent: "center",
    gap: "30px",
    background: "#f0ede9ff",
    flexWrap: "wrap",
    padding: "20px 0",
  }}
>
  {/* Card 1: Why scan websites? */}
  <div
    style={{
      background: "#fff",
      borderRadius: "3px",
      boxShadow: "0 2px 8px rgba(12, 6, 6, 0.07)",
      padding: "24px 20px",
      minWidth: "260px",
      maxWidth: "320px",
      textAlign: "center",
      flex: "1 1 260px",
      margin: "0 8px",
    }}
  >
    <div style={{ fontSize: "2rem", marginBottom: "10px" }}>🔍</div>
    <div style={{ fontWeight: "bold", fontSize: "20px", marginBottom: "8px" }}>Why scan websites?</div>
    <div style={{ fontSize: "15px", color: "#342f2fff" }}>
      Many sites hide malware, scams, or phishing. Scanning websites helps you detect threats before you click and harm your device.
    </div>
  </div>

  {/* Card 2: How to spot a malicious site */}
  <div
    style={{
      background: "#fff",
      borderRadius: "3px",
      boxShadow: "0 2px 8px rgba(0,0,0,0.07)",
      padding: "24px 20px",
      minWidth: "260px",
      maxWidth: "320px",
      textAlign: "center",
      flex: "1 1 260px",
      margin: "0 8px",
    }}
  >
    <div style={{ fontSize: "2rem", marginBottom: "10px" }}>🛑</div>
    <div style={{ fontWeight: "bold", fontSize: "20px", marginBottom: "8px" }}>Signs of a malicious website</div>
    <div style={{ fontSize: "15px", color: "#342f2fff" }}>
      Look for odd URLs, pop-ups, requests for personal info, spelling mistakes, or warnings from your browser. Be careful with any links sent via email or social media.
    </div>
  </div>

  {/* Card 3: How Scanify helps */}
  <div
    style={{
      background: "#fff",
      borderRadius: "3px",
      boxShadow: "0 2px 8px rgba(0,0,0,0.07)",
      padding: "24px 20px",
      minWidth: "260px",
      maxWidth: "320px",
      textAlign: "center",
      flex: "1 1 260px",
      margin: "0 8px",
    }}
  >
    <div style={{ fontSize: "2rem", marginBottom: "10px" }}>🤖</div>
    <div style={{ fontWeight: "bold", fontSize: "20px", marginBottom: "8px" }}>How Scanify protects you</div>
    <div style={{ fontSize: "15px", color: "#342f2fff" }}>
      Our modern tools check the URL for malicious code, phishing, and blacklists—giving you instant peace of mind.
    </div>
  </div>
</div>


<div
  style={{
    width: "100vw",
    background: "#f0ede9ff",
    display: "flex",
    justifyContent: "center",
  }}
  >
    <h2
      style={{
        color:"#bc667fff",
        fontSize: "40px",
        fontWeight: "bold",
        textAlign: "center",
        marginBottom: "15px",
        letterSpacing: "1px",

        fontFamily: "Helvetica, Arial, Tahoma, sans-serif"
      }}
      >
        How Scanify Works

      </h2>
  </div>
  <div
  style={{
    display: "flex",
    justifyContent: "center",
    gap: "30px",
    background: "#f0ede9ff",
    flexWrap: "wrap",
    padding: "20px 0",
  }}
>
 <div
  style={{
    background: "#f9f9f9",
    padding: "50px 20px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  }}
>

  <div
    style={{
      display: "flex",
      justifyContent: "flex-start",
      width: "100%",
      maxWidth: "900px",
      marginBottom: "30px",
    }}
  >
    <div
      style={{
        flex: "1",
        background: "#fff",
        padding: "25px",
        borderRadius: "3px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
      }}
    >
      <h3 style={{ fontSize: "24px", marginBottom: "10px" }}>🔗 URL Analysis</h3>
      <p style={{ fontSize: "17px", color: "#555", lineHeight: "1.7" }}>
        When you enter a website, Scanify inspects the link for suspicious
        patterns, phishing attempts, and other known signs of danger.
      </p>
    </div>
  </div>


  <div
    style={{
      display: "flex",
      justifyContent: "flex-end",
      width: "100%",
      maxWidth: "900px",
      marginBottom: "30px",
    }}
  >
    <div
      style={{
        flex: "1",
        background: "#fff",
        padding: "25px",
        borderRadius: "3px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
      }}
    >
      <h3 style={{ fontSize: "24px", marginBottom: "10px" }}>🤖 AI Detection</h3>
      <p style={{ fontSize: "17px", color: "#555", lineHeight: "1.7" }}>
        Our AI model analyzes the site’s structure and content to detect malicious code that could harm your device.
      </p>
    </div>
  </div>
\
  <div
    style={{
      display: "flex",
      justifyContent: "flex-start",
      width: "100%",
      maxWidth: "900px",
      marginBottom: "30px",
    }}
  >
    <div
      style={{
        flex: "1",
        background: "#fff",
        padding: "25px",
        borderRadius: "3px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
      }}
    >
      <h3 style={{ fontSize: "24px", marginBottom: "10px" }}>🛡️ Blacklist Check</h3>
      <p style={{ fontSize: "17px", color: "#555", lineHeight: "1.7" }}>
        Scanify compares the site against global threat databases to see if it
        has been reported for malware, phishing, or malicious activity.
      </p>
    </div>
  </div>
  <div
    style={{
      display: "flex",
      justifyContent: "flex-end",
      width: "100%",
      maxWidth: "900px",
    }}
  >
    <div
      style={{
        flex: "1",
        background: "#fff",
        padding: "25px",
        borderRadius: "3px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
      }}
    >
      <h3 style={{ fontSize: "24px", marginBottom: "10px" }}>⚡ Instant Results</h3>
      <p style={{ fontSize: "17px", color: "#555", lineHeight: "1.7" }}>
        Within seconds, Scanify provides a clear results benign, defacement, phishing, malware -
        so you can decide whether to proceed safely.
      </p>
    </div>
  </div>
</div>
</div>
<footer
        style={{
          width: "100%",
          background: "#bc667fff",
          boxShadow: "0 -2px 8px rgba(0,0,0,0.04)",
          height: "60px",
          display: "flex",
          marginTop: "40px",
          alignItems: "center",
          justifyContent: "center",
          
        }}
      >
        <div
          style={{
            width: "100%",
            maxWidth: "1100px",
            display: "flex",
            marginLeft: "20px",
             justifyContent: "space-between",
            alignItems: "center",
            padding: "0 32px",
            color: "#fef6f6ff",
            fontSize: "16px",
          }}
        >
          <div>© 2025 Scanify. All rights reserved.</div>
          <div>
            <a href="#" style={{ color: "#fff", textDecoration: "none", marginRight: "17px" }}>Privacy Policy</a>
            <a href="#" style={{ color: "#fff", textDecoration: "none" }}>Terms of Service</a>
          </div>
        </div>
      </footer>
    </>


  );

}

export default App;