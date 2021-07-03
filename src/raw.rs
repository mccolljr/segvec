use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;

#[repr(C)]
struct Header<T> {
    _len: usize,
    _cap: usize,
    _boo: PhantomData<T>,
}

impl<T> Header<T> {
    pub fn allocate(cap: usize) -> NonNull<Header<T>> {
        Self::header_with_capacity(cap)
    }

    pub unsafe fn deallocate(head: *mut Header<T>) {
        let cap = (&*head).capacity();
        std::alloc::dealloc(head as *mut u8, Self::layout(cap))
    }

    fn len(&self) -> usize {
        self._len
    }

    fn set_len(&mut self, len: usize) {
        self._len = len
    }

    fn capacity(&self) -> usize {
        self._cap
    }

    fn set_capacity(&mut self, cap: usize) {
        self._cap = cap
    }

    fn data(&self) -> *mut T {
        let padding = Self::padding();
        let header_size = mem::size_of::<Self>();
        let ptr = self as *const Self as *mut Self as *mut u8;
        unsafe {
            if padding > 0 && self.capacity() == 0 {
                // The empty header isn't well-aligned, just make an aligned one up
                NonNull::dangling().as_ptr()
            } else {
                ptr.offset((header_size + padding) as isize) as *mut T
            }
        }
    }

    fn alloc_align() -> usize {
        cmp::max(mem::align_of::<T>(), mem::align_of::<Self>())
    }

    fn layout(cap: usize) -> std::alloc::Layout {
        unsafe {
            std::alloc::Layout::from_size_align_unchecked(
                Self::alloc_size(cap),
                Self::alloc_align(),
            )
        }
    }

    fn padding() -> usize {
        let alloc_align = Self::alloc_align();
        let header_size = mem::size_of::<Self>();
        alloc_align.saturating_sub(header_size)
    }

    fn alloc_size(cap: usize) -> usize {
        // Compute "real" header size with pointer math
        let header_size = mem::size_of::<Self>();
        let elem_size = mem::size_of::<T>();
        let padding = Self::padding();
        let data_size = match elem_size.checked_mul(cap) {
            Some(data_size) if data_size <= isize::MAX as usize => data_size,
            _ => panic!("capacity overflow"),
        };
        let full_size = match data_size.checked_add(header_size + padding) {
            Some(full_size) if full_size <= isize::MAX as usize => full_size,
            _ => panic!("capacity overflow"),
        };
        full_size
    }

    fn header_with_capacity(cap: usize) -> NonNull<Header<T>> {
        debug_assert!(cap > 0);
        unsafe {
            let layout = Self::layout(cap);
            let header = std::alloc::alloc(layout) as *mut Header<T>;

            if header.is_null() {
                std::alloc::handle_alloc_error(layout)
            }

            // "Infinite" capacity for zero-sized types:
            (*header).set_capacity(cap);
            (*header).set_len(0);
            NonNull::new_unchecked(header)
        }
    }
}

#[repr(transparent)]
struct Segment<T>(NonNull<Header<T>>);

impl<T> Segment<T> {
    fn push(&mut self, val: T) {
        let header = unsafe { self.0.as_mut() };
        let len = header.len();
        let cap = header.capacity();
        assert!(len < cap, "segment overflow");
        let ptr = header.data();
        let elt_ptr = unsafe { ptr.offset(len as isize) };
        unsafe { std::ptr::write(elt_ptr, val) }
        header.set_len(len + 1);
    }

    fn pop(&mut self) -> Option<T> {
        let header = unsafe { self.0.as_mut() };
        let len = header.len();
        match len {
            0 => None,
            mut len => {
                len = len - 1;
                header.set_len(len);
                let ptr = header.data();
                let elt_ptr = unsafe { ptr.offset(len as isize) };
                Some(unsafe { std::ptr::read(elt_ptr) })
            }
        }
    }
}

impl<T> Drop for Segment<T> {
    fn drop(&mut self) {
        while let Some(_) = self.pop() {}
        unsafe { Header::deallocate(self.0.as_ptr()) }
    }
}

#[cfg(test)]
mod test_raw {
    use super::*;

    #[test]
    fn test_header() {
        #[repr(C, align(128))]
        struct Funk([u8; 3]);

        let h = Header::<Funk>::allocate(1024);
        let h_ptr = unsafe { h.as_ref() }.data();
        let h_aligned = h_ptr.align_offset(mem::align_of::<Funk>()) == 0;
        assert_eq!(mem::size_of::<Funk>(), 128);
        assert_eq!(mem::align_of::<Funk>(), 128);
        assert_eq!(Header::<Funk>::padding(), 112);
        assert_eq!(h_aligned, true);
    }

    #[test]
    fn test_segment() {
        let h = Header::<i32>::allocate(64);
        let mut s = Segment::<i32>(h);
        assert_eq!(
            std::mem::size_of::<Segment<i32>>(),
            std::mem::size_of::<usize>()
        );
        s.push(1);
        s.push(2);
        s.push(3);
        s.push(4);
        s.push(5);
        s.push(6);
        assert_eq!(s.pop(), Some(6));
        assert_eq!(s.pop(), Some(5));
        assert_eq!(s.pop(), Some(4));
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
    }
}
