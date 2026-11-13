// 检查点击抓取需要确认，并且忙碌时向用户显示错误；所有网络调用均为模拟。
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'web_cockpit.py'), 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
assert(source.includes('<option value="pick" selected>'));

const mode = { value: 'pick' };
const output = { innerText: '' };
const calls = [];
let confirmations = 0;
let allowPick = false;
let alertMessage = '';
let confirmationMessage = '';
const context = {
  window: { addEventListener() {} },
  document: { getElementById(id) { return id === 'clickMode' ? mode : output; } },
  setInterval() {},
  confirm(message) { confirmations += 1; confirmationMessage = message; return allowPick; },
  alert(message) { alertMessage = message; },
  fetch(url, options) {
    calls.push({ url, options });
    const data = url === '/api/click_pick'
      ? { success: false, error: '机械臂正在执行任务' }
      : { success: true, x_mm: 200, y_mm: -30 };
    return Promise.resolve({ json: () => Promise.resolve(data) });
  },
};
vm.createContext(context);
vm.runInContext(script, context);
context.updateTelemetry = () => {};
const click = {
  clientX: 110, clientY: 70,
  target: { getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 100 }) },
};
context.handleCanvasClick(click);
assert.strictEqual(calls.length, 0);
allowPick = true;
context.handleCanvasClick(click);
setImmediate(() => {
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0].url, '/api/click_pick');
  assert(confirmationMessage.includes('P5'));
  assert.deepStrictEqual(JSON.parse(calls[0].options.body), { norm_x: 500, norm_y: 500, target_station: 'P5' });
  assert.strictEqual(alertMessage, '机械臂正在执行任务');
  mode.value = 'preview';
  context.handleCanvasClick(click);
  setImmediate(() => {
  assert.strictEqual(calls.length, 2);
  assert.strictEqual(calls[1].url, '/api/preview_position');
  assert.deepStrictEqual(JSON.parse(calls[1].options.body), { norm_x: 500, norm_y: 500 });
  assert.strictEqual(confirmations, 2);
  assert.strictEqual(output.innerText, '预览 X:200.0 Y:-30.0 mm');
  // 每个按钮都应向偏置接口发送其标签对应的轴和 1 mm 增量。
  const buttons = [...source.matchAll(/<button\b[^>]*onclick="(nudgeCalibOffset\('([xyz])',\s*(-?1(?:\.0)?)\))"[^>]*>([XYZ][+-]1)<\/button>/g)];
  assert.strictEqual(buttons.length, 6);
  for (const [, handler, axis, delta, label] of buttons) {
    assert.strictEqual(label, axis.toUpperCase() + (Number(delta) > 0 ? '+1' : '-1'));
    const before = calls.length;
    vm.runInContext(handler, context);
    assert.strictEqual(calls.length, before + 1);
    assert.strictEqual(calls[before].url, '/api/calib_offset');
    assert.deepStrictEqual(JSON.parse(calls[before].options.body), { axis, delta: Number(delta) });
  }
  console.log('抓取确认、定位预览和 1 mm 偏置按钮检查通过');
  });
});
