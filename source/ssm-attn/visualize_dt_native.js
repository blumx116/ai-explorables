const WIDTH = 400;
const HEIGHT = 300;
const ANIMATION_DURATION = 800;
const DELAY = 1000;
const LOOP_DELAY = 2000;
const VALUE_MIN = 0;
const VALUE_MAX = 10;

const data = [
  { token: 0, delta: 0, idx: 0 },
  { token: 1, delta: 0.1, idx: 1 },
  { token: 2, delta: 0.45, idx: 2 },
  { token: 3, delta: 0.99, idx: 3 },
  { token: 1, delta: 0.1, idx: 4 },
  { token: 2, delta: 0.05, idx: 5 },
  { token: 2, delta: 0.1, idx: 6 },
  { token: 3, delta: 0.01, idx: 7 },
];

function x_axis(width, data) {
  return d3
    .scaleBand()
    .domain(data.map((d) => d.idx))
    .range([0, width])
    .padding(0.2);
}

function y_axis(height) {
  return d3
    .scaleLinear()
    .domain([VALUE_MIN, VALUE_MAX])
    .range([0, height])
    .clamp(true);
}

function linspace(low, high, count) {
  const result = [];
  const delta = high - low;

  for (var i = 0; i < count; i++) {
    result.push(low + (delta * i) / (count - 1));
  }

  return result;
}

function token_boxes(elem, data, x_axis) {
  console.log(x_axis);
  const tokenBox = elem
    .selectAll("tokens")
    .data(data)
    .enter()
    .append("g")
    .attr("opacity", (d) => 0);

  tokenBox
    .append("rect")
    .attr("x", (d) => x_axis(d.idx).toString())
    .attr("width", (d) => x_axis.bandwidth())
    .attr("y", (d) => HEIGHT + x_axis.bandwidth() / 2)
    .attr("height", (d) => x_axis.bandwidth())
    .attr("fill", (d) => tokenColorMap[d.token]);

  tokenBox
    .append("text")
    .attr("x", (d) => x_axis(d.idx) + x_axis.bandwidth() / 2)
    .attr("y", (d) => HEIGHT + x_axis.bandwidth())
    .text((d) => tokenTextMap[d.token])
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle");

  return tokenBox;
}

function chart_bars(elem, data, x_axis, cmap) {
  return elem
    .selectAll("bars")
    .data(data)
    .enter()
    .append("rect")
    .attr("x", (d) => x_axis(d.idx))
    .attr("width", (d) => x_axis.bandwidth())
    .attr("fill", (d) => cmap(d.delta))
    .on("mouseover", function (d, i) {
      obj = d3.select(this);
      obj.transition().duration(50).attr("opacity", 0.85);
    })
    .on("mouseout", function (d, i) {
      obj = d3.select(this);
      obj.transition().duration(50).attr("opacity", 1);
    });
}

function dt_plot(elem, data) {
  const total_animation_duration =
    DELAY * (data.length - 1) + ANIMATION_DURATION;

  const svg = elem
    .append("svg")
    .attr("height", HEIGHT + 200)
    .attr("width", WIDTH);

  const x_ax = x_axis(WIDTH, data);
  const y_ax = y_axis(HEIGHT);
  const cmap = d3
    .scaleLinear()
    .domain(linspace(VALUE_MIN, VALUE_MAX, cmap_range.length))
    .range(cmap_range)
    .clamp(true);

  const chart = svg.append("g");

  const labels = svg.append("g");

  const boxes = token_boxes(labels, data, x_ax);
  const bars = chart_bars(chart, data, x_ax, cmap);

  function chart_repeat() {
    bars
      .attr("y", (d) => HEIGHT)
      .attr("height", (d) => 0)
      .transition()
      .duration(ANIMATION_DURATION)
      .attr("y", function (d) {
        return HEIGHT - y_ax(d.delta);
      })
      .attr("height", function (d) {
        return y_ax(d.delta);
      })
      .delay(function (d, i) {
        return i * DELAY;
      });

    boxes
      .attr("opacity", (d) => 0)
      .transition()
      .duration(ANIMATION_DURATION)
      .delay(function (d, i) {
        return i * DELAY;
      })
      .attr("opacity", (d) => 1);

    window.visualize_dt_timeout = setTimeout(
      chart_repeat,
      total_animation_duration + LOOP_DELAY,
    );
  }

  chart_repeat();
}
