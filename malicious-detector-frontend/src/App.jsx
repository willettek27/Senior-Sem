import { useState } from "react" ;

function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

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
    <div
      style={{
        height: "100vh",
        width: "100vw",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#f0ede9ff",
      }}
    >
      <div
        style={{
          textAlign: "center",
          padding: "32px",
          borderRadius: "12px",
          background: "#fff",
          boxShadow: "0 2px 12px rgba(0,0,0,0.07)",
          marginTop: "-60px",
          minWidth: "350px",
        }}
      >
      <h1 style={{ marginBottom: "20px", fontSize: "50px", color:"#bc667fff" }}>Free website to detect malicious websites</h1>

        <input
          type="text"
          placeholder="Enter URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          style={{
            width: "300px",
            padding: "10px",
            fontSize: "18px",
            marginBottom: "16px",
          }}
          disabled={loading}
      />

      <br />

        <button
          onClick={handleScan}
          style={{
            padding: "10px 20px",
            fontSize: "16px",
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
                result === "safe"? "green": result === "Malicious" ? "red"
                : "orange",
            }}
          >
            Result: {result}
          </p>
        )}
      </div>
    </div>
  );
}
export default App;
