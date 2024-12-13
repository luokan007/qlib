class SBBStrategyEMA(SBBStrategyBase):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy with (EMA) signal.
    """

    TREND_LONG = 1  # 定义看涨趋势
    TREND_SHORT = -1  # 定义看跌趋势
    TREND_MID = 0  # 定义中性趋势

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        instruments: Union[List, str] = "csi300",
        freq: str = "day",
        trade_exchange: Exchange = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ):
        """
        初始化方法
        Parameters
        ----------
        instruments : Union[List, str], optional
            instruments of EMA signal, by default "csi300"
        freq : str, optional
            freq of EMA signal, by default "day"
            Note: `freq` may be different from `time_per_step`
        """
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")  # 如果没有提供 instruments，加载所有股票
            self.instruments = "all"
        if isinstance(instruments, str):
            self.instruments = D.instruments(instruments)  # 如果 instruments 是字符串，加载对应的股票列表
        self.freq = freq  # 设置 EMA 信号的频率
        super(SBBStrategyEMA, self).__init__(
            outer_trade_decision, level_infra, common_infra, trade_exchange=trade_exchange, **kwargs
        )  # 调用父类的初始化方法

    def _reset_signal(self):
        """
        重置信号数据
        """
        trade_len = self.trade_calendar.get_trade_len()  # 获取交易步数
        fields = ["EMA($close, 10)-EMA($close, 20)"]  # 定义 EMA 信号字段
        signal_start_time, _ = self.trade_calendar.get_step_time(trade_step=0, shift=1)  # 获取信号开始时间
        _, signal_end_time = self.trade_calendar.get_step_time(trade_step=trade_len - 1, shift=1)  # 获取信号结束时间
        signal_df = D.features(
            self.instruments, fields, start_time=signal_start_time, end_time=signal_end_time, freq=self.freq
        )  # 从数据源获取 EMA 信号数据
        signal_df.columns = ["signal"]  # 重命名列名为 "signal"
        self.signal = {}  # 初始化信号字典

        if not signal_df.empty:
            for stock_id, stock_val in signal_df.groupby(level="instrument"):
                self.signal[stock_id] = stock_val["signal"].droplevel(level="instrument")  # 将信号数据存储在字典中

    def reset_level_infra(self, level_infra):
        """
        重置层级基础设施
        - After reset the trade calendar, the signal will be changed
        """
        super().reset_level_infra(level_infra)  # 调用父类的重置方法
        self._reset_signal()  # 重置信号数据

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        """
        预测价格趋势
        """
        # 如果没有信号，返回中性趋势
        if stock_id not in self.signal:
            return self.TREND_MID
        else:
            _sample_signal = resam_ts_data(
                self.signal[stock_id],
                pred_start_time,
                pred_end_time,
                method=ts_data_last,
            )  # 从信号数据中采样特定时间范围的数据
            # 如果 EMA 信号值为 0 或 None，返回中性趋势
            if _sample_signal is None or np.isnan(_sample_signal) or _sample_signal == 0:
                return self.TREND_MID
            # 如果 EMA 信号值 > 0，返回看涨趋势
            elif _sample_signal > 0:
                return self.TREND_LONG
            # 如果 EMA 信号值 < 0，返回看跌趋势
            else:
                return self.TREND_SHORT

    def generate_trade_decision(self, execute_result=None):
        """
        生成交易决策
        """
        trade_step = self.trade_calendar.get_trade_step()  # 获取当前交易步数
        trade_len = self.trade_calendar.get_trade_len()  # 获取总交易步数

        if execute_result is not None:
            for order, _, _, _ in execute_result:
                self.trade_amount[order.stock_id] -= order.deal_amount  # 更新已成交的数量

        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)  # 获取当前交易时段的时间范围
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)  # 获取下一个交易时段的时间范围
        order_list = []  # 初始化订单列表

        for order in self.outer_trade_decision.get_decision():
            if trade_step % 2 == 0:
                _pred_trend = self._pred_price_trend(order.stock_id, pred_start_time, pred_end_time)  # 预测价格趋势
            else:
                _pred_trend = self.trade_trend[order.stock_id]  # 使用上一个时段的趋势

            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):  # 检查股票是否可交易
                if trade_step % 2 == 0:
                    self.trade_trend[order.stock_id] = _pred_trend  # 更新趋势
                continue

            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(
                stock_id=order.stock_id, start_time=order.start_time, end_time=order.end_time
            )  # 获取交易单位数量

            if _pred_trend == self.TREND_MID:
                _order_amount = None
                if _amount_trade_unit is None:
                    _order_amount = self.trade_amount[order.stock_id] / (trade_len - trade_step)  # 计算平均交易量
                else:
                    trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)  # 计算交易单位数量
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_step - 1) // (trade_len - trade_step) * _amount_trade_unit
                    )  # 计算交易量
                if order.direction == order.SELL:
                    if self.trade_amount[order.stock_id] > 1e-5 and (
                        _order_amount < 1e-5 or trade_step == trade_len - 1
                    ):
                        _order_amount = self.trade_amount[order.stock_id]  # 如果是最后一个交易步，全部卖出

                _order_amount = min(_order_amount, self.trade_amount[order.stock_id])  # 确保不超过剩余交易量

                if _order_amount > 1e-5:
                    _order = Order(
                        stock_id=order.stock_id,
                        amount=_order_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=order.direction,
                    )  # 创建订单
                    order_list.append(_order)  # 添加到订单列表

            else:
                _order_amount = None
                if _amount_trade_unit is None:
                    _order_amount = 2 * self.trade_amount[order.stock_id] / (trade_len - trade_step + 1)  # 计算加倍交易量
                else:
                    trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)  # 计算交易单位数量
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_step) // (trade_len - trade_step + 1)
                        * 2
                        * _amount_trade_unit
                    )  # 计算交易量
                if order.direction == order.SELL:
                    if self.trade_amount[order.stock_id] > 1e-5 and (
                        _order_amount < 1e-5 or trade_step == trade_len - 1
                    ):
                        _order_amount = self.trade_amount[order.stock_id]  # 如果是最后一个交易步，全部卖出

                _order_amount = min(_order_amount, self.trade_amount[order.stock_id])  # 确保不超过剩余交易量

                if _order_amount > 1e-5:
                    if trade_step % 2 == 0:
                        if (
                            _pred_trend == self.TREND_SHORT
                            and order.direction == order.SELL
                            or _pred_trend == self.TREND_LONG
                            and order.direction == order.BUY
                        ):
                            _order = Order(
                                stock_id=order.stock_id,
                                amount=_order_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=order.direction,
                            )  # 创建订单
                            order_list.append(_order)  # 添加到订单列表
                    else:
                        if (
                            _pred_trend == self.TREND_SHORT
                            and order.direction == order.BUY
                            or _pred_trend == self.TREND_LONG
                            and order.direction == order.SELL
                        ):
                            _order = Order(
                                stock_id=order.stock_id,
                                amount=_order_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=order.direction,
                            )  # 创建订单
                            order_list.append(_order)  # 添加到订单列表

            if trade_step % 2 == 0:
                self.trade_trend[order.stock_id] = _pred_trend  # 更新趋势

        return TradeDecisionWO(order_list, self)  # 返回交易决策对象