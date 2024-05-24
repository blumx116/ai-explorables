async function tensor2dToData(tensor) {
  const buffer = await tensor.buffer();
  const [y, x] = tensor.shape;
  var result = [];

  for (var i = 0; i < y; i++) {
    for (var j = 0; j < x; j++) {
      result = [
        ...result,
        {
          y: i,
          x: j,
          value: buffer.get(i, j),
          id: x * j + i,
        },
      ];
    }
  }
  return result;
}

function calculate_hw(ydim, xdim, margin) {
  // larger one is 400, other one is scaled proportionally
  const max_h = 400 - margin.top - margin.bottom;
  const max_w = 400 - margin.left - margin.right;

  return xdim > ydim
    ? [(ydim / xdim) * max_h, max_w]
    : [max_h, (xdim / ydim) * max_w];
}

function make_axes(ydim, xdim, margin) {
  const [height, width] = calculate_hw(ydim, xdim, margin);
  console.log({ height, width });

  xax = d3
    .scaleBand()
    .domain(linspace(0, xdim - 1, xdim))
    .range([0, width])
    .padding(0.01);

  yax = d3
    .scaleBand()
    .domain(linspace(0, ydim - 1, ydim))
    .range([0, height])
    .padding(0.01);

  return [yax, xax];
}

function add_tooltip(base_elem, svg) {
  var tooltip = base_elem
    .append("div")
    .style("opacity", 0)
    .attr("class", "tooltip")
    .style("background-color", "white")
    .style("border", "solid")
    .style("border-width", "2px")
    .style("border-radius", "5px")
    .style("padding", "5px");

  var mouseover = function (d) {
    tooltip.style("opacity", 1);
  };

  var mousemove = function (event, d, x, y) {
    console.log({
      event,
      d,
      x,
      y,
      mouse: d3.mouse(this),
      t: this,
      pageX: d3.event.pageX,
      pageY: d3.event.pageY,
      e: d3.event,
    });
    tooltip
      .html("Value: " + 5)
      .style("left", d3.event.screenX + "px")
      .style("top", 500 + "px");
  };
  var mouseleave = function (d) {
    tooltip.style("opacity", 0);
  };

  svg
    .on("mouseover", mouseover)
    .on("mousemove", mousemove)
    .on("mouseleave", mouseleave);
}

async function make_heatmap(elem, tensor, tokens, tokenDim) {
  var margin = { top: 30, right: 30, bottom: 30, left: 30 };
  const data = await tensor2dToData(tensor);
  const [ydim, xdim] = tensor.shape;
  const [height, width] = calculate_hw(ydim, xdim, margin);

  const [yax, xax] = make_axes(ydim, xdim, margin, tokens, tokenDim);

  const svg = elem
    .append("svg")
    .attr("height", height + margin.top + margin.bottom)
    .attr("width", width + margin.left + margin.right)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  const heatmap_cmap = d3
    .scaleLinear()
    .domain(linspace(-10, 10, cmap_range.length))
    .range(cmap_range)
    .clamp(true);

  var xfmt = d3.axisBottom(xax);
  var yfmt = d3.axisLeft(yax);

  if (tokenDim == 0) {
    yfmt = yfmt.tickFormat((d, i) => tokenTextMap[tokens[i]]);
  } else if (tokenDim == 1) {
    xfmt = xfmt.tickFormat((d, i) => tokenTextMap[tokens[i]]);
  }

  svg.append("g").style("font-size", "25px").call(xfmt);
  svg.append("g").style("font-size", "25px").call(yfmt);

  const rects = svg
    .selectAll()
    .data(data, (d) => d.id)
    .enter()
    .append("rect")
    .attr("x", (d) => xax(d.x))
    .attr("y", (d) => yax(d.y))
    .attr("width", xax.bandwidth())
    .attr("height", yax.bandwidth())
    .style("fill", (d) => heatmap_cmap(d.value));

  add_tooltip(elem, rects);
}
