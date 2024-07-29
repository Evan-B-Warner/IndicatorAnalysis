import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

// Register the required components of Chart.js
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const PriceGraph = ({ticker, dates, closePrices, MA, EMA}) => {
  const data = {
    labels: dates,
    datasets: [
      {
        label: 'Close Price',
        data: closePrices,
        fill: false,
        backgroundColor: 'rgb(75, 192, 192)',
        borderColor: 'rgba(75, 192, 192, 0.2)',
      },
      {
        label: 'MA',
        data: MA,
        fill: false,
        backgroundColor: 'rgb(192, 75, 192)',
        borderColor: 'rgba(75, 192, 192, 0.2)',
      },
      {
        label: 'EMA',
        data: EMA,
        fill: false,
        backgroundColor: 'rgb(192, 75, 75)',
        borderColor: 'rgba(75, 192, 192, 0.2)',
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: '' + ticker + ' Closing Prices',
      },
    },
  };

  return <Line data={data} options={options} />;
};

export default PriceGraph;