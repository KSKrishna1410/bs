"""
Data models for bank statement processing
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class HeaderField:
    """Represents a header field in a bank statement"""
    value: str
    method: str
    confidence: float
    raw_text: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None

@dataclass
class BankStatementHeaders:
    """Container for all header fields in a bank statement"""
    account_number: Optional[HeaderField] = None
    ifsc_code: Optional[HeaderField] = None
    bank_name: Optional[HeaderField] = None
    statement_period: Optional[HeaderField] = None
    opening_balance: Optional[HeaderField] = None
    closing_balance: Optional[HeaderField] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            field_name: (getattr(self, field_name).to_dict() if getattr(self, field_name) else None)
            for field_name in self.__annotations__
        }

@dataclass
class BankStatementTransaction:
    """Represents a single transaction in a bank statement"""
    date: datetime
    description: str
    reference: Optional[str]
    debit: Optional[float]
    credit: Optional[float]
    balance: Optional[float]
    raw_data: Dict[str, Any]

@dataclass
class ProcessedBankStatement:
    """Container for processed bank statement data"""
    headers: BankStatementHeaders
    transactions: list[BankStatementTransaction]
    metadata: Dict[str, Any] 