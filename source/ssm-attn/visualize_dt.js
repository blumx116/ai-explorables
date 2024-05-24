"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.dt_plot = exports.data = void 0;
var d3 = require("d3");
var constants = require("./constants");
var WIDTH = 400;
var HEIGHT = 300;
var ANIMATION_DURATION = 800;
var DELAY = 1000;
var LOOP_DELAY = 2000;
exports.data = [
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
        .scaleBand(data.map(function (d) { return d.idx; }), [0, width])
        .padding(0.2);
}
function y_axis(height) {
    return d3.scaleLinear([0, 1], [0, height]);
}
function linspace(low, high, count) {
    var result = [];
    var delta = high - low;
    for (var i = 0; i < count; i++) {
        result.push(low + (delta * i) / (count - 1));
    }
    return result;
}
function token_boxes(elem, data, x_axis) {
    /*
     * @param elem: DOM object to create the token boxes under
     * @param data: data of what tokens to display
     * @param x_axis: x_axis scaling/positions of the chart that the token boxes correspond to
     * 	note that only the x_axis is used b/c the boxes are all squares that infer their height from the corresponding width
     */
    // initialize invisible element with data to hold all of the token displays
    var tokenBox = elem
        .selectAll("tokens")
        .data(data)
        .enter()
        .append("g")
        .attr("opacity", function (d) { return 0; });
    // create the background squares around each token
    tokenBox
        .append("rect")
        .attr("x", function (d) { return x_axis(d.idx).toString(); })
        .attr("width", function (d) { return x_axis.bandwidth(); })
        .attr("y", function (d) { return HEIGHT + x_axis.bandwidth() / 2; })
        .attr("height", function (d) { return x_axis.bandwidth(); })
        .attr("fill", function (d) { return constants.tokenColorMap[d.token]; });
    // write the text of each token
    tokenBox
        .append("text")
        .attr("x", function (d) { return x_axis(d.idx) + x_axis.bandwidth() / 2; })
        .attr("y", function (d) { return HEIGHT + x_axis.bandwidth(); })
        .text(function (d) { return constants.tokenTextMap[d.token]; })
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
        .attr("x", function (d) { return x_axis(d.idx); })
        .attr("width", function (d) { return x_axis.bandwidth(); })
        .attr("fill", function (d) { return cmap(d.delta); });
}
function dt_plot(elem, data) {
    var total_animation_duration = DELAY * (data.length - 1) + ANIMATION_DURATION;
    var svg = elem
        .append("svg")
        .attr("height", HEIGHT + 200)
        .attr("width", WIDTH);
    var x_ax = x_axis(WIDTH, data);
    var y_ax = y_axis(HEIGHT);
    var cmap = d3.scaleLinear(linspace(0, 1, constants.cmap_range.length), constants.cmap_range);
    var chart = svg.append("g");
    var labels = svg.append("g");
    var boxes = token_boxes(labels, data, x_ax);
    var bars = chart_bars(chart, data, x_ax, cmap);
    function chart_repeat() {
        bars
            .attr("y", function (d) { return HEIGHT; })
            .attr("height", function (d) { return 0; })
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
            .attr("opacity", function (d) { return 0; })
            .transition()
            .duration(ANIMATION_DURATION)
            .delay(function (d, i) {
            return i * DELAY;
        })
            .attr("opacity", function (d) { return 1; });
        setTimeout(chart_repeat, total_animation_duration + LOOP_DELAY);
    }
    chart_repeat();
}
exports.dt_plot = dt_plot;
