# src/alert_system.py
"""
Alert system for sending notifications via multiple channels
Supports Discord, Telegram, Email, and console output
"""

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional

# Import settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings

logger = logging.getLogger(__name__)

class AlertSystem:
    """Manages alert notifications across multiple channels"""
    
    def __init__(self):
        self.settings = Settings()
        self.alert_history = []
        
        # Check which channels are configured
        self.discord_enabled = bool(self.settings.DISCORD_WEBHOOK_URL)
        self.telegram_enabled = bool(self.settings.TELEGRAM_BOT_TOKEN and self.settings.TELEGRAM_CHAT_ID)
        self.email_enabled = bool(self.settings.EMAIL_ADDRESS and self.settings.EMAIL_PASSWORD)
        
        logger.info(f"Alert channels enabled - Discord: {self.discord_enabled}, Telegram: {self.telegram_enabled}, Email: {self.email_enabled}")
    
    def send_alert(self, alert_data: Dict) -> bool:
        """Send alert to all configured channels"""
        
        success = False
        
        # Format the alert message
        message = self._format_alert(alert_data)
        
        # Always log to console
        self._console_alert(message)
        
        # Send to configured channels
        if self.discord_enabled:
            success = self._send_discord(message, alert_data) or success
        
        if self.telegram_enabled:
            success = self._send_telegram(message, alert_data) or success
        
        if self.email_enabled:
            success = self._send_email(message, alert_data) or success
        
        # Track alert history
        self._track_alert(alert_data, success)
        
        return success
    
    def _format_alert(self, alert_data: Dict) -> str:
        """Format alert data into readable message"""
        
        alert_type = alert_data.get('type', 'INFO')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Base message
        message = f"🚨 **{alert_type} ALERT** - {timestamp}\n"
        
        # Type-specific formatting
        if alert_type == 'SIGNAL':
            symbol = alert_data.get('symbol', 'Unknown')
            action = alert_data.get('action', 'Unknown')
            price = alert_data.get('price', 0)
            reasons = alert_data.get('reasons', [])
            
            message += f"\n**Symbol**: {symbol}"
            message += f"\n**Action**: {action}"
            message += f"\n**Price**: ${price:.2f}"
            
            if reasons:
                message += "\n\n**Reasons**:"
                for reason in reasons:
                    message += f"\n• {reason}"
        
        elif alert_type == 'RISK':
            symbol = alert_data.get('symbol', 'Unknown')
            risk_message = alert_data.get('message', 'Risk alert')
            
            message += f"\n**Symbol**: {symbol}"
            message += f"\n**Warning**: {risk_message}"
        
        elif alert_type == 'MARKET':
            market_data = alert_data.get('market_data', {})
            message += "\n**Market Update**"
            message += f"\n• ASX 200: {market_data.get('asx200_change', 0):+.2f}%"
            message += f"\n• Banking Sector: {market_data.get('sector_change', 0):+.2f}%"
        
        elif alert_type == 'NEWS':
            headline = alert_data.get('headline', 'News Alert')
            sentiment = alert_data.get('sentiment', 'neutral')
            
            message += f"\n**Headline**: {headline}"
            message += f"\n**Sentiment**: {sentiment}"
        
        else:
            # Generic alert
            content = alert_data.get('message', 'Alert triggered')
            message += f"\n{content}"
        
        return message
    
    def _console_alert(self, message: str):
        """Print alert to console"""
        
        print("\n" + "="*50)
        print(message.replace('**', ''))
        print("="*50 + "\n")
    
    def _send_discord(self, message: str, alert_data: Dict) -> bool:
        """Send alert to Discord webhook"""
        
        try:
            # Convert markdown to Discord format
            discord_message = message.replace('**', '**')
            
            # Create embed for rich formatting
            embed = {
                "title": f"{alert_data.get('type', 'Alert')} - {alert_data.get('symbol', 'ASX Banks')}",
                "description": discord_message,
                "color": self._get_alert_color(alert_data),
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {
                    "text": "ASX Bank Trading System"
                }
            }
            
            # Add fields for structured data
            if alert_data.get('type') == 'SIGNAL':
                embed["fields"] = [
                    {
                        "name": "Action",
                        "value": alert_data.get('action', 'N/A'),
                        "inline": True
                    },
                    {
                        "name": "Price",
                        "value": f"${alert_data.get('price', 0):.2f}",
                        "inline": True
                    }
                ]
            
            payload = {
                "embeds": [embed]
            }
            
            response = requests.post(
                self.settings.DISCORD_WEBHOOK_URL,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info("Discord alert sent successfully")
                return True
            else:
                logger.error(f"Discord alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Discord alert: {str(e)}")
            return False
    
    def _send_telegram(self, message: str, alert_data: Dict) -> bool:
        """Send alert to Telegram"""
        
        try:
            # Convert markdown to Telegram format
            telegram_message = message.replace('**', '*')
            
            url = f"https://api.telegram.org/bot{self.settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            payload = {
                'chat_id': self.settings.TELEGRAM_CHAT_ID,
                'text': telegram_message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.error(f"Telegram alert failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {str(e)}")
            return False
    
    def _send_email(self, message: str, alert_data: Dict) -> bool:
        """Send alert via email"""
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"ASX Trading Alert: {alert_data.get('type', 'Alert')} - {alert_data.get('symbol', 'Market')}"
            msg['From'] = self.settings.EMAIL_ADDRESS
            msg['To'] = self.settings.EMAIL_ADDRESS  # Send to self
            
            # Create HTML version
            html_message = self._create_html_email(message, alert_data)
            
            # Create plain text version
            text_message = message.replace('**', '')
            
            # Attach parts
            part1 = MIMEText(text_message, 'plain')
            part2 = MIMEText(html_message, 'html')
            
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            if 'gmail' in self.settings.EMAIL_ADDRESS:
                server = smtplib.SMTP('smtp.gmail.com', 587)
            else:
                # Generic SMTP settings
                server = smtplib.SMTP('smtp.mail.yahoo.com', 587)
            
            server.starttls()
            server.login(self.settings.EMAIL_ADDRESS, self.settings.EMAIL_PASSWORD)
            
            server.send_message(msg)
            server.quit()
            
            logger.info("Email alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def _create_html_email(self, message: str, alert_data: Dict) -> str:
        """Create HTML formatted email"""
        
        alert_type = alert_data.get('type', 'INFO')
        color = self._get_alert_color_hex(alert_data)
        
        html = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .alert-box {{ 
                        border: 2px solid {color}; 
                        padding: 20px; 
                        margin: 20px;
                        border-radius: 10px;
                        background-color: #f9f9f9;
                    }}
                    .alert-header {{ 
                        color: {color}; 
                        font-size: 24px; 
                        font-weight: bold;
                        margin-bottom: 10px;
                    }}
                    .alert-content {{ margin: 10px 0; }}
                    .reason-list {{ margin-left: 20px; }}
                    .footer {{ 
                        margin-top: 20px; 
                        font-size: 12px; 
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <div class="alert-box">
                    <div class="alert-header">
                        {alert_type} ALERT - ASX Bank Trading System
                    </div>
                    <div class="alert-content">
                        {self._convert_markdown_to_html(message)}
                    </div>
                    <div class="footer">
                        Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} AEST
                    </div>
                </div>
            </body>
        </html>
        """
        
        return html
    
    def _convert_markdown_to_html(self, text: str) -> str:
        """Convert markdown formatting to HTML"""
        
        # Convert bold
        text = text.replace('**', '<strong>', 1)
        text = text.replace('**', '</strong>', 1)
        
        # Convert newlines
        text = text.replace('\n', '<br>')
        
        # Convert bullet points
        text = text.replace('• ', '<li>')
        
        return text
    
    def _get_alert_color(self, alert_data: Dict) -> int:
        """Get Discord embed color based on alert type"""
        
        alert_type = alert_data.get('type', 'INFO')
        action = alert_data.get('action', '')
        
        # Color codes for Discord
        colors = {
            'SIGNAL': {
                'STRONG BUY': 0x00FF00,  # Bright green
                'BUY': 0x90EE90,         # Light green
                'SELL': 0xFFA500,        # Orange
                'STRONG SELL': 0xFF0000  # Red
            },
            'RISK': 0xFF0000,            # Red
            'MARKET': 0x0000FF,          # Blue
            'NEWS': 0xFFFF00,            # Yellow
            'INFO': 0x808080             # Gray
        }
        
        if alert_type == 'SIGNAL':
            return colors['SIGNAL'].get(action, 0x808080)
        else:
            return colors.get(alert_type, 0x808080)
    
    def _get_alert_color_hex(self, alert_data: Dict) -> str:
        """Get hex color for HTML emails"""
        
        color_int = self._get_alert_color(alert_data)
        return f"#{color_int:06X}"
    
    def _track_alert(self, alert_data: Dict, success: bool):
        """Track alert history"""
        
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_data.get('type', 'INFO'),
            'symbol': alert_data.get('symbol', ''),
            'success': success,
            'data': alert_data
        }
        
        self.alert_history.append(alert_record)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def send_batch_alerts(self, alerts: List[Dict]) -> int:
        """Send multiple alerts efficiently"""
        
        success_count = 0
        
        for alert in alerts:
            if self.send_alert(alert):
                success_count += 1
        
        return success_count
    
    def send_daily_summary(self, analysis_results: Dict, market_data: Dict):
        """Send daily summary alert"""
        
        summary_alert = {
            'type': 'DAILY_SUMMARY',
            'message': self._create_daily_summary(analysis_results, market_data)
        }
        
        self.send_alert(summary_alert)
    
    def _create_daily_summary(self, analysis_results: Dict, market_data: Dict) -> str:
        """Create daily summary message"""
        
        message = "📊 **Daily ASX Banks Summary**\n\n"
        
        # Market overview
        message += "**Market Overview**\n"
        message += f"• ASX 200: {market_data.get('ASX200', {}).get('change_percent', 0):+.2f}%\n"
        message += f"• Banking Sector: {market_data.get('trend', 'neutral')}\n\n"
        
        # Bank summaries
        message += "**Bank Analysis**\n"
        
        for symbol, analysis in analysis_results.items():
            bank_name = self.settings.get_bank_name(symbol)
            recommendation = analysis.get('recommendation', {}).get('action', 'HOLD')
            risk_score = analysis.get('risk_reward', {}).get('risk_score', 0)
            
            message += f"\n**{bank_name} ({symbol})**\n"
            message += f"• Recommendation: {recommendation}\n"
            message += f"• Risk Score: {risk_score}/100\n"
            message += f"• Price: ${analysis.get('current_price', 0):.2f}\n"
        
        return message
    
    def send_trade_confirmation(self, trade_data: Dict):
        """Send trade execution confirmation"""
        
        confirmation_alert = {
            'type': 'TRADE_CONFIRMATION',
            'symbol': trade_data.get('symbol'),
            'action': trade_data.get('action'),
            'price': trade_data.get('price'),
            'quantity': trade_data.get('quantity'),
            'message': f"Trade executed: {trade_data.get('action')} {trade_data.get('quantity')} shares of {trade_data.get('symbol')} at ${trade_data.get('price', 0):.2f}"
        }
        
        self.send_alert(confirmation_alert)
    
    def check_alert_thresholds(self, current_data: Dict) -> List[Dict]:
        """Check if any alert thresholds are triggered"""
        
        alerts = []
        thresholds = self.settings.ALERT_THRESHOLDS
        
        # Handle single analysis result vs multiple results
        if 'symbol' in current_data and 'current_price' in current_data:
            # Single analysis result - convert to expected format
            data_dict = {current_data['symbol']: current_data}
        else:
            # Multiple results - use as is
            data_dict = current_data
        
        # Price movement alerts
        for symbol, data in data_dict.items():
            if isinstance(data, dict):
                change_percent = data.get('change_percent', 0)
                
                if abs(change_percent) > thresholds['price_breakout'] * 100:
                    alerts.append({
                        'type': 'PRICE_BREAKOUT',
                        'symbol': symbol,
                        'message': f"{symbol} moved {change_percent:+.2f}% - Breakout detected",
                        'price': data.get('current_price', data.get('price', 0))
                    })
        
        return alerts
    
    def get_alert_statistics(self) -> Dict:
        """Get statistics about sent alerts"""
        
        total_alerts = len(self.alert_history)
        successful_alerts = sum(1 for alert in self.alert_history if alert['success'])
        
        alert_types = {}
        for alert in self.alert_history:
            alert_type = alert['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'successful_alerts': successful_alerts,
            'success_rate': (successful_alerts / total_alerts * 100) if total_alerts > 0 else 0,
            'alerts_by_type': alert_types,
            'last_alert': self.alert_history[-1] if self.alert_history else None
        }