import { BaseType } from "d3-selection";
import { transition } from "d3-transition";
import * as d3 from "d3";
import * as constants from "./constants";

const WIDTH = 400;
const HEIGHT = 300;
const ANIMATION_DURATION = 800;
const DELAY = 1000;
const LOOP_DELAY = 2000;

export interface DataPoint {
  token: number;
  delta: number;
  idx: number;
}

export const data: DataPoint[] = [
  { token: 0, delta: 0, idx: 0 },
  { token: 1, delta: 0.1, idx: 1 },
  { token: 2, delta: 0.45, idx: 2 },
  { token: 3, delta: 0.99, idx: 3 },
  { token: 1, delta: 0.1, idx: 4 },
  { token: 2, delta: 0.05, idx: 5 },
  { token: 2, delta: 0.1, idx: 6 },
  { token: 3, delta: 0.01, idx: 7 },
];

function x_axis(width: number, data: DataPoint[]): d3.ScaleBand<number> {
  return d3
    .scaleBand<number>()
    .domain(data.map((d) => d.idx))
    .range([0, width])
    .padding(0.2);
}

function y_axis(height: number): d3.ScaleLinear<number, number, never> {
  return d3.scaleLinear([-1, 10], [0, height]).clamp(true);
}

function linspace(low: number, high: number, count: number): number[] {
  const result: number[] = [];
  const delta: number = high - low;

  for (var i = 0; i < count; i++) {
    result.push(low + (delta * i) / (count - 1));
  }

  return result;
}

function token_boxes(
  elem: d3.Selection<SVGGElement, unknown, any, any>,
  data: DataPoint[],
  x_axis: d3.ScaleBand<number>,
): d3.Selection<SVGGElement, DataPoint, any, any> {
  /*
   * @param elem: DOM object to create the token boxes under
   * @param data: data of what tokens to display
   * @param x_axis: x_axis scaling/positions of the chart that the token boxes correspond to
   * 	note that only the x_axis is used b/c the boxes are all squares that infer their height from the corresponding width
   */

  // initialize invisible element with data to hold all of the token displays
  const tokenBox: d3.Selection<SVGGElement, DataPoint, any, any> = elem
    .selectAll("tokens")
    .data(data)
    .enter()
    .append("g")
    .attr("opacity", (d) => 0);

  // create the background squares around each token
  tokenBox
    .append("rect")
    .attr("x", (d) => x_axis(d.idx)!.toString())
    .attr("width", (d) => x_axis.bandwidth())
    .attr("y", (d) => HEIGHT + x_axis.bandwidth() / 2)
    .attr("height", (d) => x_axis.bandwidth())
    .attr("fill", (d) => constants.tokenColorMap[d.token]);

  // write the text of each token
  tokenBox
    .append("text")
    .attr("x", (d) => x_axis(d.idx)! + x_axis.bandwidth() / 2)
    .attr("y", (d) => HEIGHT + x_axis.bandwidth())
    .text((d) => constants.tokenTextMap[d.token])
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle");

  return tokenBox;
}

function chart_bars(
  elem: d3.Selection<SVGGElement, unknown, any, any>,
  data: DataPoint[],
  x_axis: d3.ScaleBand<number>,
  cmap: d3.ScaleLinear<string, string>,
): d3.Selection<SVGRectElement, DataPoint, SVGGElement, any> {
  return elem
    .selectAll("bars")
    .data(data)
    .enter()
    .append("rect")
    .attr("x", (d) => x_axis(d.idx)!)
    .attr("width", (d) => x_axis.bandwidth())
    .attr("fill", (d) => cmap(d.delta));
}

export function dt_plot(
  elem: d3.Selection<BaseType, unknown, HTMLElement, any>,
  data: DataPoint[],
): void {
  const total_animation_duration: number =
    DELAY * (data.length - 1) + ANIMATION_DURATION;

  const svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any> = elem
    .append("svg")
    .attr("height", HEIGHT + 200)
    .attr("width", WIDTH);

  const x_ax: d3.ScaleBand<number> = x_axis(WIDTH, data);
  const y_ax: d3.ScaleLinear<number, number, never> = y_axis(HEIGHT);
  const cmap: d3.ScaleLinear<string, string> = d3
    .scaleLinear(
      linspace(0, 1, constants.cmap_range.length),
      constants.cmap_range,
    )
    .clamp(true);

  const chart: d3.Selection<SVGGElement, unknown, HTMLElement, any> =
    svg.append("g");

  const labels: d3.Selection<SVGGElement, unknown, HTMLElement, any> =
    svg.append("g");

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

    setTimeout(chart_repeat, total_animation_duration + LOOP_DELAY);
  }

  chart_repeat();
}
