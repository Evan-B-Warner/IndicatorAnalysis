import './App.css';
import React, { useState } from 'react';
import axios from 'axios';
import PriceGraph from './components/PriceGraph';

function App() {
  const [isOverlayDropdownOpen, setIsOverlayDropdownOpen] = useState(false);
  const [selectedOverlays, setSelectedOverlays] = useState([]);
  const [ticker, setTicker] = useState("");
  const [dates, setDates] = useState([]);
  const [closePrices, setClosePrices] = useState([]);
  const [MA, setMA] = useState([]);
  const [EMA, setEMA] = useState([]);

  const overlayIndicators = [
    { id: 1, label: 'MA' },
    { id: 2, label: 'EMA' }
  ];

  const pullTickerData = async () => {
    const response = await axios.get(`http://127.0.0.1:5000/api/price?ticker=${ticker}`);
    setDates(response.data.dates);
    setClosePrices(response.data.closePrices);
  }

  const toggleDropdown = () => {
    setIsOverlayDropdownOpen(!isOverlayDropdownOpen);
  }

  const handleCheckboxChange = (id) => {
    setSelectedOverlays((prevSelected) =>
      prevSelected.includes(id)
        ? prevSelected.filter((item) => item !== id)
        : [...prevSelected, id]
    );
    
    console.log(selectedOverlays);
    selectedOverlays.map((overlay) => {

    })

  };

  return (
    <div>
      <div>
        <form onSubmit={e => e.preventDefault()}>
          <label>Ticker:</label>
          <input 
            id="tickerInput"
            type="text"
            value={ticker}
            onChange={e => setTicker(e.target.value)}>
          </input>
          <button className="submitTicker" onClick={() => pullTickerData()}>Submit</button>
        </form>
      </div>

      <div style={{ position: 'relative', display: 'inline-block' }}>
      <button onClick={toggleDropdown}>
        Add Overlays {selectedOverlays.length > 0 && `(${selectedOverlays.length})`}
      </button>
      {isOverlayDropdownOpen && (
        <div style={{ position: 'absolute', border: '1px solid #ccc', backgroundColor: '#fff', zIndex: 1 }}>
          {overlayIndicators.map((option) => (
            <label key={option.id} style={{ display: 'block', padding: '8px' }}>
              <input
                type="checkbox"
                checked={selectedOverlays.includes(option.id)}
                onChange={() => handleCheckboxChange(option.id)}
              />
              {option.label}
            </label>
          ))}
        </div>
        )}
      </div>

      <div>
        <PriceGraph ticker={ticker} dates={dates} closePrices={closePrices} MA={MA} EMA={EMA}/>
      </div>
    </div>
  );
}

export default App;
